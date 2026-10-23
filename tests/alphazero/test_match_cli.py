"""Tests for scripts/alphazero/match.py checkpoint loading.

Coverage:
  - Each network's architecture is detected from its own checkpoint, so
    two differently-sized networks (e.g. Bonanza v2 10x128 vs v1 4x64)
    can be loaded into one match without any size flags.
  - Explicit size flags still override detection, and a wrong override
    fails loudly instead of silently loading the wrong network.
  - The loaded networks can actually play a (tiny) match against each
    other through run_match.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "alphazero"))

from make_random_checkpoint import make_random_checkpoint  # noqa: E402
from match import _load_network  # noqa: E402

from shogi_dojo.alphazero.evaluation import run_match  # noqa: E402

SMALL = dict(num_blocks=1, num_filters=8, value_head_hidden=8)
LARGER = dict(num_blocks=2, num_filters=16, value_head_hidden=24)


def _arch_of(net) -> dict:
    return dict(
        num_blocks=len(net.tower),
        num_filters=net.stem_conv.out_channels,
        value_head_hidden=net.value_fc1.out_features,
    )


@pytest.fixture
def two_checkpoints(tmp_path):
    small = tmp_path / "small.pt"
    larger = tmp_path / "larger.pt"
    make_random_checkpoint(str(small), seed=0, **SMALL)
    make_random_checkpoint(str(larger), seed=1, **LARGER)
    return str(small), str(larger)


class TestLoadNetwork:
    def test_detects_each_architecture(self, two_checkpoints):
        small, larger = two_checkpoints
        assert _arch_of(_load_network(small)) == SMALL
        assert _arch_of(_load_network(larger)) == LARGER

    def test_weights_match_checkpoint(self, two_checkpoints):
        _, larger = two_checkpoints
        net = _load_network(larger)
        saved = torch.load(larger, weights_only=True)["model_state_dict"]
        for key, value in net.state_dict().items():
            assert torch.equal(value, saved[key])

    def test_loaded_in_eval_mode(self, two_checkpoints):
        assert not _load_network(two_checkpoints[0]).training

    def test_matching_override_is_accepted(self, two_checkpoints):
        net = _load_network(two_checkpoints[1], num_blocks=2, num_filters=16)
        assert _arch_of(net) == LARGER

    def test_wrong_override_fails_loudly(self, two_checkpoints):
        with pytest.raises(RuntimeError):     # load_state_dict size mismatch
            _load_network(two_checkpoints[1], num_filters=8)

    def test_different_sizes_can_play_a_match(self, two_checkpoints):
        small, larger = two_checkpoints
        result = run_match(
            _load_network(larger), _load_network(small),
            num_games=2, num_simulations=2, move_limit=6,
        )
        assert result.total_games == 2
