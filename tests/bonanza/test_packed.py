"""Tests for the packed (pre-processed) Bonanza dataset.

Coverage goals:
  - pack_samples / unpack_batch: lossless round trip — decoded tensors are
    bit-for-bit identical to replay_game's output (inputs, legal masks,
    targets, value targets), including hand planes and drop moves.
  - pack_samples rejects inputs that break the compression assumptions.
  - preprocess_csa_files: shard layout, sample order, resume (skips
    finished shards, redoes missing ones), settings-mismatch refusal,
    unreadable files counted, multi-process output identical to
    single-process.
  - PackedDataset: len, single and batched access across shard
    boundaries, out-of-range errors, version/config checks, ignores
    half-written .tmp shards, pickling drops memmaps, DataLoader with
    worker processes.
  - BonanzaTrainer trains on a PackedDataset.
  - scripts/bonanza/preprocess_kifu.py CLI and train_bonanza.py --packed.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from shogi_dojo.alphazero.encoding import NUM_INPUT_CHANNELS, NUM_MOVE_PLANES
from shogi_dojo.bonanza.dataset import parse_csa_string, replay_game
from shogi_dojo.bonanza.packed import (
    CONFIG_FILE,
    FORMAT_VERSION,
    PLANE_BYTES,
    PackedDataset,
    pack_samples,
    preprocess_csa_files,
    unpack_batch,
)
from shogi_dojo.bonanza.trainer import BonanzaTrainer
from shogi_dojo.nnue.constants import NUM_FILES, NUM_RANKS

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "bonanza"))

from preprocess_kifu import main as preprocess_main  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic games
# ---------------------------------------------------------------------------

_START = """\
V2.2
N+TestBlack
N-TestWhite
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA *\x20
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  *\x20
P5 *  *  *  *  *  *  *  *  *\x20
P6 *  *  *  *  *  *  *  *  *\x20
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 * +KA *  *  *  *  * +HI *\x20
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
+
"""

# Bishop exchange then a bishop drop: exercises captures, promotion,
# non-empty hands (hand planes) and a drop move (drop policy planes).
# White resigns at the end, so Black wins.
GAME_BISHOP_TRADE = _START + """\
+7776FU
-3334FU
+8822UM
-3122GI
+0045KA
%TORYO
"""

# Black resigns after three moves → White wins.
GAME_SHORT = _START + """\
+7776FU
-3334FU
+2726FU
%TORYO
"""

# Interrupted game → no winner → value target 0.
GAME_UNFINISHED = _START + """\
+2726FU
-8384FU
%CHUDAN
"""

GAMES = [GAME_BISHOP_TRADE, GAME_SHORT, GAME_UNFINISHED]


def _write_games(tmp_path: Path, n: int) -> list[str]:
    """Write n CSA files cycling through GAMES; return their paths."""
    paths = []
    for i in range(n):
        p = tmp_path / f"game_{i:03d}.csa"
        p.write_text(GAMES[i % len(GAMES)])
        paths.append(str(p))
    return paths


def _replay_all(paths: list[str], min_ply: int = 1) -> list[dict]:
    """Reference samples: what the unpacked pipeline produces, in file order."""
    samples = []
    for p in paths:
        for game in parse_csa_string(Path(p).read_text()):
            samples.extend(replay_game(game, min_ply=min_ply))
    return samples


def _assert_sample_equal(got: dict, ref: dict) -> None:
    assert torch.equal(got["input"], ref["input"])
    assert torch.equal(got["legal_mask"], ref["legal_mask"])
    assert int(got["target"]) == ref["target"]
    assert float(got["value_target"]) == ref["value_target"]


def _unpack_all(arrays: dict) -> dict:
    off = arrays["legal_offsets"]
    lists = [arrays["legal_indices"][off[i]:off[i + 1]] for i in range(len(off) - 1)]
    return unpack_batch(
        arrays["planes"], arrays["scalars"], arrays["target"], arrays["value"], lists,
    )


# ---------------------------------------------------------------------------
# pack_samples / unpack_batch
# ---------------------------------------------------------------------------


class TestPackRoundTrip:
    def test_round_trip_is_lossless(self, tmp_path):
        ref = _replay_all(_write_games(tmp_path, 3))
        batch = _unpack_all(pack_samples(ref))
        assert batch["input"].shape == (len(ref), NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
        assert batch["legal_mask"].shape == (len(ref), NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)
        assert batch["input"].dtype == torch.float32
        assert batch["target"].dtype == torch.int64
        assert batch["legal_mask"].dtype == torch.bool
        assert batch["value_target"].dtype == torch.float32
        for i, r in enumerate(ref):
            _assert_sample_equal({k: v[i] for k, v in batch.items()}, r)

    def test_synthetic_games_cover_hands_drops_and_all_outcomes(self, tmp_path):
        # Guard that the fixtures actually exercise the interesting cases,
        # so the round-trip test above is meaningful.
        ref = _replay_all(_write_games(tmp_path, 3))
        hand_planes = torch.stack([s["input"][28:42] for s in ref])
        assert hand_planes.abs().sum() > 0, "no sample has pieces in hand"
        drop_plane_base = 132
        assert any(s["target"] // 81 >= drop_plane_base for s in ref), "no drop move"
        assert {s["value_target"] for s in ref} == {-1.0, 0.0, 1.0}

    def test_storage_dtypes_and_shapes(self, tmp_path):
        ref = _replay_all(_write_games(tmp_path, 1))
        arrays = pack_samples(ref)
        n = len(ref)
        assert arrays["planes"].shape == (n, PLANE_BYTES)
        assert arrays["planes"].dtype == np.uint8
        assert arrays["scalars"].shape == (n, 17)
        assert arrays["target"].dtype == np.uint16
        assert arrays["value"].dtype == np.int8
        assert arrays["legal_offsets"].shape == (n + 1,)
        assert arrays["legal_offsets"][-1] == arrays["legal_indices"].shape[0]
        expected_legal = sum(int(s["legal_mask"].sum()) for s in ref)
        assert arrays["legal_indices"].shape[0] == expected_legal

    def test_empty_input(self):
        arrays = pack_samples([])
        assert arrays["planes"].shape == (0, PLANE_BYTES)
        assert arrays["legal_offsets"].tolist() == [0]
        assert _unpack_all(arrays)["input"].shape[0] == 0

    def test_rejects_non_binary_piece_plane(self, tmp_path):
        ref = _replay_all(_write_games(tmp_path, 1))[:1]
        ref[0] = dict(ref[0], input=ref[0]["input"].clone())
        ref[0]["input"][0, 0, 0] = 0.5
        with pytest.raises(ValueError, match="0.0 / 1.0"):
            pack_samples(ref)

    def test_rejects_non_constant_scalar_plane(self, tmp_path):
        ref = _replay_all(_write_games(tmp_path, 1))[:1]
        ref[0] = dict(ref[0], input=ref[0]["input"].clone())
        ref[0]["input"][44, 3, 3] += 0.25
        with pytest.raises(ValueError, match="constant"):
            pack_samples(ref)


# ---------------------------------------------------------------------------
# preprocess_csa_files
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_shards_and_sample_order(self, tmp_path):
        files = _write_games(tmp_path, 5)
        out = tmp_path / "packed"
        summary = preprocess_csa_files(files, out, files_per_shard=2)
        assert summary.shards_built == 3          # 2 + 2 + 1 files
        assert summary.shards_skipped == 0
        assert summary.games_built == 5
        assert summary.failed_files == 0

        ds = PackedDataset(out)
        ref = _replay_all(files)
        assert len(ds) == len(ref) == summary.samples_built
        assert ds.num_shards == 3
        for i, r in enumerate(ref):
            _assert_sample_equal(ds[i], r)

        meta = json.loads((out / "shard_00002" / "meta.json").read_text())
        assert meta["num_files"] == 1
        assert (out / "shard_00000" / "files.txt").read_text().splitlines() == files[:2]

    def test_min_ply_is_applied_and_recorded(self, tmp_path):
        files = _write_games(tmp_path, 3)
        out = tmp_path / "packed"
        preprocess_csa_files(files, out, min_ply=3)
        ds = PackedDataset(out)
        assert ds.config["min_ply"] == 3
        assert len(ds) == len(_replay_all(files, min_ply=3))

    def test_resume_skips_finished_and_rebuilds_missing(self, tmp_path):
        files = _write_games(tmp_path, 4)
        out = tmp_path / "packed"
        preprocess_csa_files(files, out, files_per_shard=1)
        before = PackedDataset(out)[0]["input"].clone()

        shutil.rmtree(out / "shard_00002")
        summary = preprocess_csa_files(files, out, files_per_shard=1)
        assert summary.shards_built == 1
        assert summary.shards_skipped == 3
        assert len(PackedDataset(out)) == len(_replay_all(files))
        assert torch.equal(PackedDataset(out)[0]["input"], before)

    def test_refuses_different_settings_in_same_directory(self, tmp_path):
        files = _write_games(tmp_path, 2)
        out = tmp_path / "packed"
        preprocess_csa_files(files, out, min_ply=1)
        with pytest.raises(ValueError, match="different settings"):
            preprocess_csa_files(files, out, min_ply=10)

    def test_unreadable_file_is_counted_not_fatal(self, tmp_path):
        files = _write_games(tmp_path, 2) + [str(tmp_path / "missing.csa")]
        out = tmp_path / "packed"
        summary = preprocess_csa_files(files, out)
        assert summary.failed_files == 1
        assert summary.games_built == 2
        meta = json.loads((out / "shard_00000" / "meta.json").read_text())
        assert meta["failed_files"] == [files[-1]]

    def test_multiprocess_matches_single_process(self, tmp_path):
        files = _write_games(tmp_path, 6)
        preprocess_csa_files(files, tmp_path / "a", files_per_shard=2, num_workers=1)
        preprocess_csa_files(files, tmp_path / "b", files_per_shard=2, num_workers=2)
        for shard in ("shard_00000", "shard_00001", "shard_00002"):
            for name in ("planes", "scalars", "target", "value", "legal_offsets", "legal_indices"):
                a = np.load(tmp_path / "a" / shard / f"{name}.npy")
                b = np.load(tmp_path / "b" / shard / f"{name}.npy")
                np.testing.assert_array_equal(a, b)

    def test_progress_callback_called_per_shard(self, tmp_path):
        files = _write_games(tmp_path, 3)
        seen = []
        preprocess_csa_files(files, tmp_path / "p", files_per_shard=1, progress=seen.append)
        assert sorted(r["shard_id"] for r in seen) == [0, 1, 2]

    def test_rejects_bad_files_per_shard(self, tmp_path):
        with pytest.raises(ValueError):
            preprocess_csa_files([], tmp_path / "p", files_per_shard=0)


# ---------------------------------------------------------------------------
# PackedDataset
# ---------------------------------------------------------------------------


@pytest.fixture
def packed(tmp_path):
    """A 3-shard packed dataset plus its reference samples."""
    files = _write_games(tmp_path, 6)
    out = tmp_path / "packed"
    preprocess_csa_files(files, out, files_per_shard=2)
    return out, _replay_all(files)


class TestPackedDataset:
    def test_batched_access_across_shards_in_random_order(self, packed):
        out, ref = packed
        ds = PackedDataset(out)
        idx = np.random.default_rng(0).permutation(len(ds)).tolist()
        batch = ds.__getitems__(idx)
        assert batch["input"].shape[0] == len(idx)
        for pos, i in enumerate(idx):
            _assert_sample_equal({k: v[pos] for k, v in batch.items()}, ref[i])

    def test_out_of_range_raises(self, packed):
        ds = PackedDataset(packed[0])
        with pytest.raises(IndexError):
            ds[len(ds)]
        with pytest.raises(IndexError):
            ds.__getitems__([0, -1])

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PackedDataset(tmp_path)

    def test_format_version_mismatch_raises(self, packed):
        out, _ = packed
        cfg = json.loads((out / CONFIG_FILE).read_text())
        cfg["format_version"] = FORMAT_VERSION + 1
        (out / CONFIG_FILE).write_text(json.dumps(cfg))
        with pytest.raises(ValueError, match="format_version"):
            PackedDataset(out)

    def test_no_shards_raises(self, tmp_path):
        preprocess_csa_files([], tmp_path / "empty")
        with pytest.raises(ValueError, match="No non-empty shards"):
            PackedDataset(tmp_path / "empty")

    def test_ignores_half_written_tmp_shard(self, packed):
        out, ref = packed
        shutil.copytree(out / "shard_00000", out / "shard_00099.tmp")
        assert len(PackedDataset(out)) == len(ref)

    def test_pickle_drops_open_memmaps(self, packed):
        ds = PackedDataset(packed[0])
        ds[0]  # forces the memmaps open
        clone = pickle.loads(pickle.dumps(ds))
        assert clone._shards is None
        assert len(pickle.dumps(ds)) < 10_000  # no array data inside
        _assert_sample_equal(clone[1], packed[1][1])

    def test_dataloader_with_workers(self, packed):
        out, ref = packed
        ds = PackedDataset(out)
        loader = DataLoader(
            ds, batch_size=4, shuffle=True, num_workers=2, collate_fn=ds.collate_fn,
        )
        seen = 0
        for batch in loader:
            assert batch["input"].shape[1:] == (NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
            assert batch["legal_mask"].shape[1:] == (NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)
            # The professional's move must always be legal.
            flat = batch["legal_mask"].view(batch["target"].shape[0], -1)
            assert flat.gather(1, batch["target"][:, None]).all()
            seen += batch["target"].shape[0]
        assert seen == len(ref)


# ---------------------------------------------------------------------------
# Trainer and CLI integration
# ---------------------------------------------------------------------------


def _tiny_config(checkpoint_dir: str) -> dict:
    return {
        "model": {"num_blocks": 1, "num_filters": 8, "value_head_hidden": 8},
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "lambda_value": 0.5,
            "checkpoint_dir": checkpoint_dir,
            "save_every": 2,
        },
    }


class TestIntegration:
    def test_trainer_accepts_packed_dataset(self, packed, tmp_path):
        ds = PackedDataset(packed[0])
        ckpt_dir = tmp_path / "ckpt"
        trainer = BonanzaTrainer(_tiny_config(str(ckpt_dir)))
        before = [p.detach().clone() for p in trainer.network.parameters()]
        trainer.train(ds)
        assert (ckpt_dir / "bonanza_epoch_0002.pt").exists()
        after = list(trainer.network.parameters())
        assert any(not torch.equal(a, b) for a, b in zip(before, after))

    def test_preprocess_cli(self, tmp_path):
        files = _write_games(tmp_path, 3)
        listing = tmp_path / "files.txt"
        listing.write_text("\n".join(files) + "\n")
        out = tmp_path / "packed"
        preprocess_main([
            "--data-list", str(listing), "--output", str(out),
            "--files-per-shard", "2", "--workers", "1", "--min-ply", "2",
        ])
        ds = PackedDataset(out)
        assert ds.num_shards == 2
        assert len(ds) == len(_replay_all(files, min_ply=2))

    def test_train_bonanza_packed_flag(self, packed, tmp_path, monkeypatch):
        sys.path.insert(0, _REPO_ROOT)
        import train_bonanza

        cfg_path = tmp_path / "cfg.yaml"
        import yaml
        cfg_path.write_text(yaml.safe_dump(_tiny_config(str(tmp_path / "ckpt"))))
        monkeypatch.setattr(sys, "argv", [
            "train_bonanza.py", "--config", str(cfg_path), "--packed", str(packed[0]),
        ])
        train_bonanza.main()
        assert (tmp_path / "ckpt" / "bonanza_epoch_0002.pt").exists()
