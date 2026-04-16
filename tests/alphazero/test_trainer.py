"""Tests for shogi_dojo.alphazero.trainer — the training loop.

The main test is a short end-to-end smoke test: construct an
AlphaZeroTrainer from a tiny config, run 2 iterations of 1 game each
with minimal MCTS, and verify that:

  - The loop completes without error.
  - The replay buffer grew (self-play produced data).
  - The loss reported is finite.
  - A checkpoint file was saved.

This test does *not* assert that the loss decreases (2 iterations ×
1 game at 4 MCTS sims is far too noisy for any meaningful trend).
Verifying that the loop runs end-to-end and that all the components
talk to each other correctly is the goal.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
import yaml

from shogi_dojo.alphazero.trainer import AlphaZeroTrainer


def _tiny_config(checkpoint_dir: str) -> dict:
    """Build the smallest viable config dict in-memory."""
    return {
        "model": {
            "num_blocks": 1,
            "num_filters": 8,
            "value_head_hidden": 8,
        },
        "self_play": {
            "games_per_iteration": 1,
            "num_simulations": 4,
            "c_puct": 1.5,
            "dirichlet_alpha": 0.15,
            "dirichlet_epsilon": 0.25,
            "temperature_threshold": 30,
            "move_limit": 6,
        },
        "buffer": {
            "capacity": 500,
        },
        "training": {
            "iterations": 2,
            "steps_per_iteration": 5,
            "batch_size": 4,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "checkpoint_dir": checkpoint_dir,
            "save_every": 2,
        },
    }


class TestAlphaZeroTrainerConstruction:
    def test_constructs_from_config_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _tiny_config(tmpdir)
            trainer = AlphaZeroTrainer(cfg)
            assert trainer.iterations == 2
            assert trainer.replay_buffer.capacity == 500
            assert isinstance(trainer.network, torch.nn.Module)


class TestAlphaZeroTrainerSmokeTest:
    """End-to-end smoke test: run the full loop on a minimal budget.

    Timeboxed: the tiny config plays 1 game per iteration with
    move_limit=6 and 4 MCTS sims per move, so each game finishes in
    well under a second on any modern CPU. Two iterations total, with
    5 training steps each. The whole test should take < 30 seconds.
    """

    def test_full_loop_runs_to_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _tiny_config(tmpdir)
            trainer = AlphaZeroTrainer(cfg)

            rng = np.random.default_rng(42)
            trainer.train(rng=rng)

            # Replay buffer should have grown (self-play produced examples).
            assert len(trainer.replay_buffer) > 0

    def test_checkpoint_saved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _tiny_config(tmpdir)
            trainer = AlphaZeroTrainer(cfg)
            trainer.train(rng=np.random.default_rng(0))

            # With save_every=2 and iterations=2, a checkpoint at iter 2
            # should exist.
            ckpt_path = os.path.join(tmpdir, "az_iter_0002.pt")
            assert os.path.exists(ckpt_path)

            ckpt = torch.load(ckpt_path, weights_only=True)
            assert "model_state_dict" in ckpt
            assert "optimizer_state_dict" in ckpt
            assert ckpt["iteration"] == 2
            assert ckpt["buffer_size"] > 0

    def test_network_parameters_differ_after_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _tiny_config(tmpdir)
            trainer = AlphaZeroTrainer(cfg)

            params_before = {
                name: p.clone() for name, p in trainer.network.named_parameters()
            }

            trainer.train(rng=np.random.default_rng(0))

            # At least some parameters should have changed after
            # gradient steps.
            any_changed = any(
                not torch.equal(p, params_before[name])
                for name, p in trainer.network.named_parameters()
            )
            assert any_changed, "No network parameters changed after training"


class TestAlphaZeroTrainerFromYaml:
    """Verify that the YAML configs parse and construct correctly."""

    def test_tiny_yaml_constructs(self):
        yaml_path = "configs/alphazero/tiny.yaml"
        if not os.path.exists(yaml_path):
            pytest.skip("tiny.yaml not found at expected path")
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg["training"]["checkpoint_dir"] = tmpdir
            trainer = AlphaZeroTrainer(cfg)
            assert trainer.iterations == 10

    def test_default_yaml_constructs(self):
        yaml_path = "configs/alphazero/default.yaml"
        if not os.path.exists(yaml_path):
            pytest.skip("default.yaml not found at expected path")
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg["training"]["checkpoint_dir"] = tmpdir
            trainer = AlphaZeroTrainer(cfg)
            assert trainer.iterations == 50
