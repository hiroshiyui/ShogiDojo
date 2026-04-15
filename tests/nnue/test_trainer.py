"""Tests for shogi_dojo.trainer — loss functions, Trainer init, and checkpointing."""

import os
import tempfile

import torch

from shogi_dojo.trainer import Trainer, result_loss, sigmoid_loss


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class TestSigmoidLoss:
    """Tests for the score-matching loss: MSE(sigmoid(pred), sigmoid(target))."""

    def test_zero_loss_when_equal(self):
        pred = torch.tensor([1.0, -1.0, 0.0])
        loss = sigmoid_loss(pred, pred)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)

    def test_positive_when_different(self):
        pred = torch.tensor([1.0, 0.0])
        target = torch.tensor([-1.0, 2.0])
        loss = sigmoid_loss(pred, target)
        assert loss.item() > 0.0

    def test_symmetric(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([-1.0, 0.5])
        assert torch.isclose(sigmoid_loss(a, b), sigmoid_loss(b, a))

    def test_output_is_scalar(self):
        pred = torch.tensor([0.5, 1.5])
        target = torch.tensor([0.0, 1.0])
        loss = sigmoid_loss(pred, target)
        assert loss.dim() == 0


class TestResultLoss:
    """Tests for the game-result loss: MSE(sigmoid(pred), result)."""

    def test_zero_loss_when_perfect(self):
        # sigmoid(0) = 0.5, so predicting 0.0 should give zero loss for result=0.5 (draw).
        pred = torch.tensor([0.0])
        target = torch.tensor([0.5])
        loss = result_loss(pred, target)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)

    def test_win_wants_high_score(self):
        # For result=1.0 (win), higher predicted score → lower loss.
        target = torch.tensor([1.0])
        loss_high = result_loss(torch.tensor([5.0]), target)
        loss_low = result_loss(torch.tensor([-5.0]), target)
        assert loss_high < loss_low

    def test_loss_wants_low_score(self):
        # For result=0.0 (loss), lower predicted score → lower loss.
        target = torch.tensor([0.0])
        loss_neg = result_loss(torch.tensor([-5.0]), target)
        loss_pos = result_loss(torch.tensor([5.0]), target)
        assert loss_neg < loss_pos

    def test_output_is_scalar(self):
        loss = result_loss(torch.tensor([0.0, 1.0]), torch.tensor([0.5, 1.0]))
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# Trainer — helper to build a minimal config
# ---------------------------------------------------------------------------

def _minimal_config(tmp_dir: str, **overrides) -> dict:
    """Build the smallest valid config for Trainer.__init__."""
    train_path = os.path.join(tmp_dir, "train.sfen")
    # Write a tiny training file so DataLoader can be created.
    with open(train_path, "w") as f:
        f.write(
            "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\t100\t1.0\n"
        )

    cfg = {
        "data": {
            "train_path": train_path,
            "batch_size": 1,
            "num_workers": 0,
        },
        "model": {
            "ft_out_dim": 8,
            "l1_dim": 4,
            "l2_dim": 4,
        },
        "training": {
            "epochs": 2,
            "lr": 1e-3,
            "lr_scheduler": "cosine",
            "lr_step_size": 50,
            "lr_gamma": 0.5,
            "weight_decay": 0.0,
            "lambda_score": 1.0,
            "lambda_result": 0.5,
            "score_scaling": 600.0,
            "clip_grad_norm": 1.0,
            "checkpoint_dir": os.path.join(tmp_dir, "ckpts"),
            "save_every": 1,
            "early_stopping_patience": 0,
        },
    }
    # Apply any overrides (flat keys into the training sub-dict).
    for k, v in overrides.items():
        cfg["training"][k] = v
    return cfg


# ---------------------------------------------------------------------------
# Trainer init
# ---------------------------------------------------------------------------

class TestTrainerInit:
    """Verify that the Trainer initialises correctly from config."""

    def test_creates_model_and_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(tmp)
            trainer = Trainer(cfg)
            assert trainer.model is not None
            assert trainer.optimizer is not None

    def test_cosine_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(tmp, lr_scheduler="cosine")
            trainer = Trainer(cfg)
            assert trainer.scheduler is not None

    def test_step_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(tmp, lr_scheduler="step")
            trainer = Trainer(cfg)
            assert trainer.scheduler is not None

    def test_no_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(tmp, lr_scheduler="none")
            trainer = Trainer(cfg)
            assert trainer.scheduler is None

    def test_early_stopping_patience(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(tmp, early_stopping_patience=10)
            trainer = Trainer(cfg)
            assert trainer.early_stopping_patience == 10


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

class TestCheckpointing:
    """Verify checkpoint saving and that saved files contain expected keys."""

    def test_save_checkpoint_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(tmp)
            trainer = Trainer(cfg)
            os.makedirs(trainer.checkpoint_dir, exist_ok=True)
            trainer._save_checkpoint(1)
            path = os.path.join(trainer.checkpoint_dir, "epoch_0001.pt")
            assert os.path.isfile(path)

    def test_checkpoint_contains_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(tmp)
            trainer = Trainer(cfg)
            os.makedirs(trainer.checkpoint_dir, exist_ok=True)
            trainer._save_checkpoint(5)
            ckpt = torch.load(
                os.path.join(trainer.checkpoint_dir, "epoch_0005.pt"),
                weights_only=False,
            )
            assert "epoch" in ckpt
            assert "model_state_dict" in ckpt
            assert "optimizer_state_dict" in ckpt
            assert ckpt["epoch"] == 5

    def test_save_best_model_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(tmp)
            trainer = Trainer(cfg)
            os.makedirs(trainer.checkpoint_dir, exist_ok=True)
            trainer._save_best_model(3, 0.123)
            path = os.path.join(trainer.checkpoint_dir, "best_model.pt")
            assert os.path.isfile(path)
            ckpt = torch.load(path, weights_only=False)
            assert ckpt["best_epoch"] == 3
            assert abs(ckpt["best_loss"] - 0.123) < 1e-6


# ---------------------------------------------------------------------------
# Training loop (minimal smoke test)
# ---------------------------------------------------------------------------

class TestTrainLoop:
    """Smoke-test that the training loop runs without error on tiny data."""

    def test_train_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(tmp, epochs=2, save_every=1)
            trainer = Trainer(cfg)
            trainer.train()
            # Should have saved checkpoints.
            assert os.path.isfile(
                os.path.join(trainer.checkpoint_dir, "epoch_0002.pt")
            )
            # Best model should always be saved.
            assert os.path.isfile(
                os.path.join(trainer.checkpoint_dir, "best_model.pt")
            )

    def test_early_stopping(self):
        """With patience=1, training should stop before reaching max epochs."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(
                tmp, epochs=100, early_stopping_patience=1, save_every=100
            )
            trainer = Trainer(cfg)
            # Force best_loss to a tiny value so every epoch counts as
            # "no improvement", triggering early stop after 1 epoch.
            trainer.best_loss = -1.0
            trainer.train()
            # Should have stopped well before epoch 100.
            assert not os.path.isfile(
                os.path.join(trainer.checkpoint_dir, "epoch_0100.pt")
            )

    def test_save_every_at_final_epoch(self):
        """Even if save_every doesn't divide epochs, the final epoch is saved."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(tmp, epochs=3, save_every=5)
            trainer = Trainer(cfg)
            trainer.train()
            # Epoch 3 is the final epoch, should be saved even though 3 % 5 != 0.
            assert os.path.isfile(
                os.path.join(trainer.checkpoint_dir, "epoch_0003.pt")
            )

    def test_weights_change_after_training(self):
        """Model weights should be different after training (optimizer stepped)."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(tmp, epochs=2, save_every=2)
            trainer = Trainer(cfg)

            # Snapshot initial weights.
            initial_weights = {
                name: param.clone()
                for name, param in trainer.model.named_parameters()
            }

            trainer.train()

            # At least some weights should have changed.
            any_changed = False
            for name, param in trainer.model.named_parameters():
                if not torch.equal(param, initial_weights[name]):
                    any_changed = True
                    break
            assert any_changed, "model weights should change after training"

    def test_no_val_path_uses_train_loss(self):
        """When val_path is not set, training should still complete and track
        best model using training loss."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(tmp, epochs=2, save_every=1)
            # Ensure no val_path.
            cfg["data"].pop("val_path", None)
            trainer = Trainer(cfg)
            trainer.train()
            assert os.path.isfile(
                os.path.join(trainer.checkpoint_dir, "best_model.pt")
            )

    def test_checkpoint_can_be_loaded(self):
        """A saved checkpoint contains loadable model weights."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(tmp, epochs=1, save_every=1)
            trainer = Trainer(cfg)
            trainer.train()

            # Load the checkpoint and verify it restores model state.
            ckpt_path = os.path.join(trainer.checkpoint_dir, "epoch_0001.pt")
            ckpt = torch.load(ckpt_path, weights_only=False)

            new_model = trainer.model.__class__(
                ft_input_dim=trainer.model.ft.weight.shape[0],
                ft_out_dim=trainer.model.ft.weight.shape[1],
                l1_dim=trainer.model.l1.out_features,
                l2_dim=trainer.model.l2.out_features,
            )
            new_model.load_state_dict(ckpt["model_state_dict"])
            # Verify loaded weights match saved weights.
            for (n1, p1), (n2, p2) in zip(
                trainer.model.named_parameters(), new_model.named_parameters()
            ):
                assert torch.equal(p1.cpu(), p2.cpu()), f"weight mismatch for {n1}"

    def test_lr_changes_with_cosine_scheduler(self):
        """Cosine scheduler should decrease the learning rate over epochs."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _minimal_config(tmp, epochs=5, save_every=5, lr_scheduler="cosine")
            trainer = Trainer(cfg)
            initial_lr = trainer.optimizer.param_groups[0]["lr"]
            trainer.train()
            final_lr = trainer.optimizer.param_groups[0]["lr"]
            assert final_lr < initial_lr, "LR should decrease with cosine scheduler"
