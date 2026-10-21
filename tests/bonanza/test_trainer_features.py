"""Tests for BonanzaTrainer's optional training features.

Coverage goals:
  - make_lr_lambda: warmup ramp, constant / cosine / step shapes,
    validation of bad configs.
  - Optimizer selection (adam / adamw / sgd) and bad names.
  - LR scheduler inside train(): reaches the expected final LR, state is
    checkpointed and restored on resume (resumed run == uninterrupted
    run), old checkpoints without scheduler state still resume,
    streaming datasets need an explicit total_steps.
  - AMP: disabled with a warning on CPU; on CUDA trains, keeps losses
    finite and checkpoints the grad-scaler state.
  - Validation: evaluate() metrics and side-effect freedom,
    bonanza_best.pt tracking, best_val_loss restored on resume.
  - log_every_steps progress logging.
  - PackedDataset.split / shard_ids subsets.
  - train_bonanza.py --val-fraction and configs/bonanza/v2.yaml.
"""

from __future__ import annotations

import logging
import math
import sys

import pytest
import torch
import yaml

from shogi_dojo.bonanza.dataset import StreamingSupervisedDataset
from shogi_dojo.bonanza.packed import PackedDataset, preprocess_csa_files
from shogi_dojo.bonanza.trainer import BonanzaTrainer, make_lr_lambda
from tests.bonanza.test_packed import _REPO_ROOT, _write_games


def _cfg(checkpoint_dir, **training) -> dict:
    cfg = {
        "model": {"num_blocks": 1, "num_filters": 8, "value_head_hidden": 8},
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "lambda_value": 0.5,
            "checkpoint_dir": str(checkpoint_dir),
            "save_every": 1,
            "num_workers": 0,
        },
    }
    cfg["training"].update(training)
    return cfg


@pytest.fixture
def packed_dir(tmp_path):
    """6 games in 6 shards (one game per shard) → ~20 samples."""
    out = tmp_path / "packed"
    preprocess_csa_files(_write_games(tmp_path, 6), out, files_per_shard=1)
    return out


def _lr(trainer: BonanzaTrainer) -> float:
    return trainer.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# make_lr_lambda
# ---------------------------------------------------------------------------


class TestLrLambda:
    def test_constant_with_warmup(self):
        f = make_lr_lambda({"type": "constant", "warmup_steps": 4}, total_steps=None)
        assert [f(s) for s in range(6)] == [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]

    def test_cosine_shape(self):
        f = make_lr_lambda(
            {"type": "cosine", "warmup_steps": 10, "min_lr_ratio": 0.1}, total_steps=110,
        )
        assert f(10) == pytest.approx(1.0)
        assert f(60) == pytest.approx(0.55)   # halfway: (1 + 0.1) / 2
        assert f(110) == pytest.approx(0.1)
        assert f(500) == pytest.approx(0.1)   # clamped after the end
        values = [f(s) for s in range(10, 111)]
        assert all(a >= b for a, b in zip(values, values[1:]))  # monotone decay

    def test_step_decay(self):
        f = make_lr_lambda(
            {"type": "step", "milestones": [0.5, 0.75], "gamma": 0.1}, total_steps=100,
        )
        assert f(49) == 1.0
        assert f(50) == pytest.approx(0.1)
        assert f(75) == pytest.approx(0.01)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="lr_schedule.type"):
            make_lr_lambda({"type": "linear"}, total_steps=10)

    def test_decay_without_total_steps_raises(self):
        with pytest.raises(ValueError, match="total_steps"):
            make_lr_lambda({"type": "cosine"}, total_steps=None)

    def test_bad_schedule_rejected_at_trainer_construction(self, tmp_path):
        with pytest.raises(ValueError, match="lr_schedule.type"):
            BonanzaTrainer(_cfg(tmp_path, lr_schedule={"type": "nope"}))


# ---------------------------------------------------------------------------
# Optimizer selection
# ---------------------------------------------------------------------------


class TestOptimizer:
    @pytest.mark.parametrize("name, cls", [
        (None, torch.optim.Adam),          # default stays v1's Adam
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("sgd", torch.optim.SGD),
    ])
    def test_selection(self, tmp_path, name, cls):
        extra = {} if name is None else {"optimizer": name}
        trainer = BonanzaTrainer(_cfg(tmp_path, **extra))
        assert type(trainer.optimizer) is cls

    def test_sgd_uses_nesterov_momentum(self, tmp_path):
        trainer = BonanzaTrainer(_cfg(tmp_path, optimizer="sgd"))
        group = trainer.optimizer.param_groups[0]
        assert group["momentum"] == 0.9 and group["nesterov"]

    def test_unknown_optimizer_raises(self, tmp_path):
        with pytest.raises(ValueError, match="optimizer"):
            BonanzaTrainer(_cfg(tmp_path, optimizer="lion"))


# ---------------------------------------------------------------------------
# LR scheduler inside train()
# ---------------------------------------------------------------------------


class TestSchedulerInTraining:
    def test_cosine_reaches_min_lr_at_end(self, tmp_path, packed_dir):
        ds = PackedDataset(packed_dir)
        cfg = _cfg(tmp_path / "ckpt", lr_schedule={"type": "cosine", "min_lr_ratio": 0.05})
        trainer = BonanzaTrainer(cfg)
        trainer.train(ds)
        assert _lr(trainer) == pytest.approx(1e-3 * 0.05)

    def test_default_schedule_keeps_lr_constant(self, tmp_path, packed_dir):
        trainer = BonanzaTrainer(_cfg(tmp_path / "ckpt"))
        trainer.train(PackedDataset(packed_dir))
        assert _lr(trainer) == pytest.approx(1e-3)

    def test_resume_matches_uninterrupted_run(self, tmp_path, packed_dir):
        ds = PackedDataset(packed_dir)
        # Pin total_steps so the 1-epoch "interrupted" run follows the
        # same 3-epoch schedule as the full run.
        sched = {
            "type": "cosine", "warmup_steps": 2, "min_lr_ratio": 0.01,
            "total_steps": 3 * math.ceil(len(ds) / 4),
        }

        full = BonanzaTrainer(_cfg(tmp_path / "a", epochs=3, lr_schedule=sched))
        full.train(ds)

        part = BonanzaTrainer(_cfg(tmp_path / "b", epochs=1, lr_schedule=sched))
        part.train(ds)
        mid_lr = _lr(part)

        resumed = BonanzaTrainer(_cfg(tmp_path / "b", epochs=3, lr_schedule=sched))
        resumed.resume_from(str(tmp_path / "b" / "bonanza_epoch_0001.pt"))
        resumed.train(ds)

        assert mid_lr > _lr(full)  # sanity: LR actually decayed after epoch 1
        assert resumed.scheduler.last_epoch == full.scheduler.last_epoch
        assert _lr(resumed) == pytest.approx(_lr(full))

    def test_checkpoint_without_scheduler_state_still_resumes(self, tmp_path, packed_dir):
        ds = PackedDataset(packed_dir)
        trainer = BonanzaTrainer(_cfg(tmp_path, epochs=1))
        trainer.train(ds)
        path = tmp_path / "bonanza_epoch_0001.pt"
        ckpt = torch.load(path, weights_only=True)
        for key in ("scheduler_state_dict", "scaler_state_dict", "best_val_loss"):
            ckpt.pop(key, None)  # what a v1 checkpoint looks like
        torch.save(ckpt, path)

        resumed = BonanzaTrainer(_cfg(tmp_path, epochs=2, lr_schedule={"type": "cosine"}))
        resumed.resume_from(str(path))
        assert resumed.best_val_loss == math.inf
        resumed.train(ds)  # must not raise

    def test_streaming_decay_needs_total_steps(self, tmp_path):
        files = _write_games(tmp_path, 2)
        ds = StreamingSupervisedDataset(csa_files=files, buffer_size=2, seed=0)
        trainer = BonanzaTrainer(_cfg(tmp_path / "c", lr_schedule={"type": "cosine"}))
        with pytest.raises(ValueError, match="total_steps"):
            trainer.train(ds)

        trainer = BonanzaTrainer(
            _cfg(tmp_path / "d", lr_schedule={"type": "cosine", "total_steps": 10}),
        )
        trainer.train(ds)
        assert trainer.scheduler.last_epoch > 0


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------


class TestAmp:
    def test_amp_disabled_on_cpu_with_warning(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with caplog.at_level(logging.WARNING):
            trainer = BonanzaTrainer(_cfg(tmp_path, amp=True))
        assert trainer.amp is False
        assert not trainer.scaler.is_enabled()
        assert "amp requested" in caplog.text

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_amp_training_on_cuda(self, tmp_path, packed_dir):
        trainer = BonanzaTrainer(_cfg(tmp_path, amp=True))
        assert trainer.amp is True
        before = [p.detach().clone() for p in trainer.network.parameters()]
        trainer.train(PackedDataset(packed_dir))
        assert any(not torch.equal(a, b) for a, b in zip(before, trainer.network.parameters()))
        assert all(torch.isfinite(p).all() for p in trainer.network.parameters())
        ckpt = torch.load(tmp_path / "bonanza_epoch_0002.pt", weights_only=True)
        assert "scaler_state_dict" in ckpt

        resumed = BonanzaTrainer(_cfg(tmp_path, amp=True, epochs=3))
        resumed.resume_from(str(tmp_path / "bonanza_epoch_0002.pt"))
        assert resumed.scaler.get_scale() == trainer.scaler.get_scale()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_evaluate_metrics_and_no_side_effects(self, tmp_path, packed_dir):
        ds = PackedDataset(packed_dir)
        trainer = BonanzaTrainer(_cfg(tmp_path))
        trainer.network.train()
        before = {k: v.clone() for k, v in trainer.network.state_dict().items()}

        m = trainer.evaluate(ds)

        assert m["samples"] == len(ds)
        assert 0.0 <= m["accuracy"] <= 1.0
        assert m["loss"] == pytest.approx(m["policy_loss"] + 0.5 * m["value_loss"], rel=1e-4)
        assert trainer.network.training  # mode restored
        after = trainer.network.state_dict()
        # BatchNorm running stats must not move in eval mode.
        assert all(torch.equal(before[k], after[k]) for k in before)

    def test_evaluate_is_sample_weighted(self, tmp_path, packed_dir):
        # Batch sizes that do not divide the dataset must give the same
        # answer, because the last partial batch is weighted by its size.
        ds = PackedDataset(packed_dir)
        trainer = BonanzaTrainer(_cfg(tmp_path))
        trainer.batch_size = len(ds)
        whole = trainer.evaluate(ds)
        trainer.batch_size = 3
        chunked = trainer.evaluate(ds)
        for k in ("loss", "policy_loss", "value_loss", "accuracy"):
            assert chunked[k] == pytest.approx(whole[k], rel=1e-4, abs=1e-6)

    def test_best_checkpoint_and_metrics_saved(self, tmp_path, packed_dir):
        train_ds, val_ds = PackedDataset(packed_dir).split(0.34, seed=0)
        trainer = BonanzaTrainer(_cfg(tmp_path, epochs=3))
        trainer.train(train_ds, val_dataset=val_ds)

        best = torch.load(tmp_path / "bonanza_best.pt", weights_only=False)
        assert best["val_metrics"]["samples"] == len(val_ds)
        assert best["val_metrics"]["loss"] == pytest.approx(trainer.best_val_loss)
        epoch3 = torch.load(tmp_path / "bonanza_epoch_0003.pt", weights_only=False)
        assert "val_metrics" in epoch3
        assert epoch3["best_val_loss"] == pytest.approx(trainer.best_val_loss)

    def test_no_val_dataset_means_no_best_file(self, tmp_path, packed_dir):
        BonanzaTrainer(_cfg(tmp_path)).train(PackedDataset(packed_dir))
        assert not (tmp_path / "bonanza_best.pt").exists()

    def test_best_val_loss_restored_on_resume(self, tmp_path, packed_dir):
        train_ds, val_ds = PackedDataset(packed_dir).split(0.34)
        trainer = BonanzaTrainer(_cfg(tmp_path, epochs=1))
        trainer.train(train_ds, val_dataset=val_ds)
        resumed = BonanzaTrainer(_cfg(tmp_path, epochs=2))
        resumed.resume_from(str(tmp_path / "bonanza_epoch_0001.pt"))
        assert resumed.best_val_loss == pytest.approx(trainer.best_val_loss)


def test_log_every_steps(tmp_path, packed_dir, caplog):
    ds = PackedDataset(packed_dir)
    with caplog.at_level(logging.INFO, logger="shogi_dojo.bonanza.trainer"):
        BonanzaTrainer(_cfg(tmp_path, epochs=1, batch_size=2, log_every_steps=2)).train(ds)
    step_lines = [r.getMessage() for r in caplog.records if r.getMessage().lstrip().startswith("step ")]
    assert len(step_lines) == math.ceil(len(ds) / 2) // 2
    assert "lr=" in step_lines[0] and "samples/s" in step_lines[0]


# ---------------------------------------------------------------------------
# PackedDataset.split
# ---------------------------------------------------------------------------


class TestSplit:
    def test_disjoint_cover_and_deterministic(self, packed_dir):
        full = PackedDataset(packed_dir)
        train, val = full.split(0.34, seed=0)
        assert set(train.shard_ids).isdisjoint(val.shard_ids)
        assert sorted(train.shard_ids + val.shard_ids) == full.shard_ids
        assert len(train) + len(val) == len(full)
        assert val.num_shards == 2  # round(0.34 × 6)
        assert full.split(0.34, seed=0)[1].shard_ids == val.shard_ids

    def test_seed_changes_the_split(self, packed_dir):
        full = PackedDataset(packed_dir)
        splits = {tuple(full.split(0.5, seed=s)[1].shard_ids) for s in range(10)}
        assert len(splits) > 1

    def test_at_least_one_shard_each_side(self, packed_dir):
        full = PackedDataset(packed_dir)
        assert full.split(0.01)[1].num_shards == 1
        assert full.split(0.99)[0].num_shards == 1

    def test_bad_arguments(self, tmp_path, packed_dir):
        with pytest.raises(ValueError):
            PackedDataset(packed_dir).split(0.0)
        with pytest.raises(ValueError):
            PackedDataset(packed_dir).split(1.0)
        one = tmp_path / "one"
        preprocess_csa_files(_write_games(tmp_path, 1), one)
        with pytest.raises(ValueError, match="at least 2 shards"):
            PackedDataset(one).split(0.5)

    def test_shard_ids_subset(self, packed_dir):
        sub = PackedDataset(packed_dir, shard_ids=[1, 4])
        assert sub.shard_ids == [1, 4]
        assert len(sub) == len(PackedDataset(packed_dir, [1])) + len(PackedDataset(packed_dir, [4]))


# ---------------------------------------------------------------------------
# CLI and shipped config
# ---------------------------------------------------------------------------


class TestCliAndConfig:
    def _run(self, monkeypatch, argv):
        sys.path.insert(0, _REPO_ROOT)
        import train_bonanza
        monkeypatch.setattr(sys, "argv", ["train_bonanza.py", *argv])
        train_bonanza.main()

    def test_val_fraction_trains_and_saves_best(self, tmp_path, packed_dir, monkeypatch):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(_cfg(tmp_path / "ckpt")))
        self._run(monkeypatch, [
            "--config", str(cfg_path), "--packed", str(packed_dir), "--val-fraction", "0.34",
        ])
        assert (tmp_path / "ckpt" / "bonanza_best.pt").exists()

    def test_val_fraction_requires_packed(self, tmp_path, monkeypatch):
        files = _write_games(tmp_path, 1)
        with pytest.raises(SystemExit):
            self._run(monkeypatch, ["--data", *files, "--val-fraction", "0.1"])

    def test_v2_config_constructs(self, tmp_path):
        with open(f"{_REPO_ROOT}/configs/bonanza/v2.yaml") as f:
            cfg = yaml.safe_load(f)
        cfg["training"]["checkpoint_dir"] = str(tmp_path)
        trainer = BonanzaTrainer(cfg)
        assert type(trainer.optimizer) is torch.optim.AdamW
        assert trainer.lr_schedule_cfg["type"] == "cosine"
        n_params = sum(p.numel() for p in trainer.network.parameters())
        assert 2_500_000 < n_params < 4_000_000
