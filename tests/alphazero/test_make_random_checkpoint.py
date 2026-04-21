"""Tests for the random-checkpoint helper.

Coverage:
- File created with the expected payload keys.
- Loadable into ``AlphaZeroNetwork`` without modification (i.e. the
  layout matches what trainer-produced checkpoints use).
- ``--seed`` produces deterministic weights; omitting it does not.
- Custom architecture flags are honored end-to-end.
- Saved file is compatible with ``torch.load(weights_only=True)``,
  matching how ``match.py`` loads checkpoints.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

# Make scripts/alphazero/ importable so we can call the helper directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "alphazero"))

from make_random_checkpoint import make_random_checkpoint  # noqa: E402

from shogi_dojo.alphazero.network import AlphaZeroNetwork  # noqa: E402


TINY = dict(num_blocks=1, num_filters=8, value_head_hidden=8)


class TestMakeRandomCheckpoint:
    def test_file_created_with_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rand.pt")
            make_random_checkpoint(path, **TINY)
            assert os.path.exists(path)

            ckpt = torch.load(path, weights_only=True, map_location="cpu")
            assert set(ckpt.keys()) == {
                "epoch", "global_step", "model_state_dict", "optimizer_state_dict",
            }
            assert ckpt["epoch"] == 0
            assert ckpt["global_step"] == 0
            assert ckpt["optimizer_state_dict"] == {}

    def test_loadable_into_alphazero_network(self):
        # Same load pattern as scripts/alphazero/match.py — must not
        # require any fix-ups or custom keys.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rand.pt")
            make_random_checkpoint(path, **TINY)

            net = AlphaZeroNetwork(**TINY)
            ckpt = torch.load(path, weights_only=True, map_location="cpu")
            net.load_state_dict(ckpt["model_state_dict"])
            net.eval()

            # Quick functional check: forward pass returns the expected
            # two-tuple of correctly-shaped tensors.
            x = torch.randn(2, 45, 9, 9)
            with torch.no_grad():
                policy, value = net(x)
            assert policy.shape == (2, 139, 9, 9)
            assert value.shape == (2, 1)

    def test_seed_makes_weights_reproducible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "a.pt")
            path_b = os.path.join(tmpdir, "b.pt")
            make_random_checkpoint(path_a, seed=42, **TINY)
            make_random_checkpoint(path_b, seed=42, **TINY)

            sd_a = torch.load(path_a, weights_only=True, map_location="cpu")["model_state_dict"]
            sd_b = torch.load(path_b, weights_only=True, map_location="cpu")["model_state_dict"]
            assert set(sd_a.keys()) == set(sd_b.keys())
            for k in sd_a:
                assert torch.equal(sd_a[k], sd_b[k]), f"Tensor {k} diverged across seeded runs"

    def test_different_seeds_produce_different_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "a.pt")
            path_b = os.path.join(tmpdir, "b.pt")
            make_random_checkpoint(path_a, seed=0, **TINY)
            make_random_checkpoint(path_b, seed=1, **TINY)

            sd_a = torch.load(path_a, weights_only=True, map_location="cpu")["model_state_dict"]
            sd_b = torch.load(path_b, weights_only=True, map_location="cpu")["model_state_dict"]
            # At least one parameter tensor must differ — same seed
            # would have caught all of them, different seeds should
            # diverge on every Kaiming-initialized layer.
            assert any(
                not torch.equal(sd_a[k], sd_b[k])
                for k in sd_a
                if sd_a[k].numel() > 0
            )

    def test_custom_architecture_flags_honored(self):
        # Saved state dict shapes must reflect the requested arch, not
        # the defaults — otherwise match.py would refuse to load with
        # a size-mismatch error.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rand.pt")
            make_random_checkpoint(
                path, num_blocks=2, num_filters=16, value_head_hidden=32,
            )
            ckpt = torch.load(path, weights_only=True, map_location="cpu")
            sd = ckpt["model_state_dict"]
            # Stem conv: (out=num_filters=16, in=45, 3, 3)
            assert sd["stem_conv.weight"].shape == (16, 45, 3, 3)
            # Value head hidden layer: (out=value_head_hidden=32, in=81)
            assert sd["value_fc1.weight"].shape == (32, 81)
            # Two residual blocks → keys for tower.0.* and tower.1.*
            tower_indices = {k.split(".")[1] for k in sd if k.startswith("tower.")}
            assert tower_indices == {"0", "1"}
