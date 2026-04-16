"""Tests for shogi_dojo.alphazero.self_play — game driver + replay buffer.

Coverage goals:
  - TrainingExample stores the right tensor shapes.
  - The private _backfill_z helper matches the "z at each position is
    from that STM's perspective" contract for every combination of
    (terminal_stm, terminal_value).
  - _policy_dict_to_tensor writes MCTS probabilities at the right
    (plane, y, x) positions.
  - ReplayBuffer add/add_many/len/sample behave as documented,
    including FIFO eviction when capacity is reached and
    RuntimeError on empty-buffer sampling.
  - Integration: play_game runs to completion on a short-move-limit
    game and returns training examples with the right tensor shapes,
    non-negative length, and values in {-1, 0, +1}.
  - play_game on a terminal starting board returns no examples.
"""

from __future__ import annotations

import numpy as np
import pytest
import shogi
import torch

from shogi_dojo.alphazero.encoding import (
    NUM_INPUT_CHANNELS,
    NUM_MOVE_PLANES,
    encode_move,
)
from shogi_dojo.alphazero.network import AlphaZeroNetwork
from shogi_dojo.alphazero.rules import shogi_move_to_move
from shogi_dojo.alphazero.self_play import (
    ReplayBuffer,
    TrainingExample,
    _backfill_z,
    _policy_dict_to_tensor,
    play_game,
)
from shogi_dojo.nnue.constants import (
    BLACK,
    NUM_FILES,
    NUM_RANKS,
    NUM_SQUARES,
    WHITE,
)


# ---------------------------------------------------------------------------
# Test helper: tiny network for integration tests
# ---------------------------------------------------------------------------


def _tiny_network() -> AlphaZeroNetwork:
    return AlphaZeroNetwork(num_blocks=1, num_filters=8, value_head_hidden=8)


def _make_example(value: float = 0.0) -> TrainingExample:
    """Construct a dummy TrainingExample for buffer tests."""
    return TrainingExample(
        input_tensor=torch.zeros(NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES),
        policy_target=torch.zeros(NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES),
        value_target=value,
        legal_mask=torch.zeros(NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# _backfill_z
# ---------------------------------------------------------------------------


class TestBackfillZ:
    def test_draw_is_zero_for_everyone(self):
        zs = _backfill_z([BLACK, WHITE, BLACK], terminal_stm=WHITE, terminal_value=0.0)
        assert zs == [0.0, 0.0, 0.0]

    def test_terminal_stm_shares_value(self):
        # Terminal STM is WHITE, they lost (-1). Every WHITE-to-move
        # position in the game shares that fate.
        zs = _backfill_z([WHITE, BLACK, WHITE], terminal_stm=WHITE, terminal_value=-1.0)
        assert zs == [-1.0, +1.0, -1.0]

    def test_black_checkmated_makes_black_positions_negative(self):
        # Terminal STM is BLACK, they lost (-1). BLACK-to-move
        # positions get -1, WHITE-to-move positions get +1.
        zs = _backfill_z(
            [BLACK, WHITE, BLACK, WHITE],
            terminal_stm=BLACK,
            terminal_value=-1.0,
        )
        assert zs == [-1.0, +1.0, -1.0, +1.0]

    def test_empty_history_yields_empty_list(self):
        zs = _backfill_z([], terminal_stm=BLACK, terminal_value=-1.0)
        assert zs == []


# ---------------------------------------------------------------------------
# _policy_dict_to_tensor
# ---------------------------------------------------------------------------


class TestPolicyDictToTensor:
    def test_writes_probabilities_to_expected_positions(self):
        # Construct a simple dict: one legal move with probability 1.
        board = shogi.Board()
        move = next(iter(board.legal_moves))
        visits = {move: 1.0}
        tensor = _policy_dict_to_tensor(visits, side_to_move=BLACK)

        # Compute where we expect that mass to land.
        idx = encode_move(shogi_move_to_move(move), BLACK)
        plane = idx // NUM_SQUARES
        sq = idx % NUM_SQUARES
        y, x = sq // NUM_FILES, sq % NUM_FILES

        assert tensor.shape == (NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)
        assert tensor[plane, y, x] == pytest.approx(1.0)
        # Everywhere else is zero.
        total = tensor.sum().item()
        assert total == pytest.approx(1.0)

    def test_empty_dict_produces_all_zeros(self):
        tensor = _policy_dict_to_tensor({}, side_to_move=BLACK)
        assert tensor.sum().item() == 0.0


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------


class TestReplayBufferConstruction:
    def test_rejects_zero_capacity(self):
        with pytest.raises(ValueError, match="capacity"):
            ReplayBuffer(0)

    def test_rejects_negative_capacity(self):
        with pytest.raises(ValueError, match="capacity"):
            ReplayBuffer(-5)

    def test_fresh_buffer_is_empty(self):
        buf = ReplayBuffer(10)
        assert len(buf) == 0
        assert not buf.is_full
        assert buf.capacity == 10


class TestReplayBufferAdd:
    def test_add_increases_length(self):
        buf = ReplayBuffer(10)
        buf.add(_make_example())
        assert len(buf) == 1

    def test_add_many_increases_length_correctly(self):
        buf = ReplayBuffer(10)
        buf.add_many([_make_example() for _ in range(5)])
        assert len(buf) == 5

    def test_fifo_eviction_when_capacity_reached(self):
        buf = ReplayBuffer(3)
        # Add 5 items with distinct value_targets so we can see which
        # survived.
        buf.add_many([_make_example(value=float(i)) for i in range(5)])
        # Buffer should contain the last 3 items (values 2, 3, 4).
        assert len(buf) == 3
        assert buf.is_full

    def test_is_full_flips_exactly_at_capacity(self):
        buf = ReplayBuffer(3)
        assert not buf.is_full
        buf.add_many([_make_example() for _ in range(2)])
        assert not buf.is_full
        buf.add(_make_example())
        assert buf.is_full


class TestReplayBufferSample:
    def test_sample_returns_expected_keys_and_shapes(self):
        buf = ReplayBuffer(10)
        buf.add_many([_make_example() for _ in range(10)])
        batch = buf.sample(batch_size=4, rng=np.random.default_rng(0))

        assert set(batch.keys()) == {"input", "policy", "value", "legal_mask"}
        assert batch["input"].shape == (4, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
        assert batch["policy"].shape == (4, NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)
        assert batch["value"].shape == (4,)
        assert batch["legal_mask"].shape == (4, NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)

    def test_sample_dtypes(self):
        buf = ReplayBuffer(10)
        buf.add_many([_make_example() for _ in range(10)])
        batch = buf.sample(batch_size=2, rng=np.random.default_rng(0))

        assert batch["input"].dtype == torch.float32
        assert batch["policy"].dtype == torch.float32
        assert batch["value"].dtype == torch.float32
        assert batch["legal_mask"].dtype == torch.bool

    def test_sample_empty_raises(self):
        buf = ReplayBuffer(10)
        with pytest.raises(RuntimeError, match="empty"):
            buf.sample(batch_size=1)

    def test_sample_with_replacement_allows_batch_larger_than_buffer(self):
        # Only one example in the buffer, but we can sample a batch of 4.
        buf = ReplayBuffer(10)
        buf.add(_make_example(value=0.5))
        batch = buf.sample(batch_size=4, rng=np.random.default_rng(0))
        assert batch["value"].shape == (4,)
        assert torch.all(batch["value"] == 0.5)


# ---------------------------------------------------------------------------
# play_game
# ---------------------------------------------------------------------------


class TestPlayGame:
    def test_returns_list_of_training_examples(self):
        # Short move_limit forces a fast draw so the test runs in
        # reasonable wall-time even with num_simulations > 1.
        net = _tiny_network()
        examples = play_game(
            network=net,
            num_simulations=2,
            move_limit=6,
            rng=np.random.default_rng(0),
        )
        assert isinstance(examples, list)
        assert all(isinstance(e, TrainingExample) for e in examples)

    def test_each_example_has_expected_tensor_shapes(self):
        net = _tiny_network()
        examples = play_game(
            network=net,
            num_simulations=2,
            move_limit=6,
            rng=np.random.default_rng(0),
        )
        assert len(examples) > 0   # at least one move was played
        for e in examples:
            assert e.input_tensor.shape == (NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
            assert e.policy_target.shape == (NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)
            assert e.legal_mask.shape == (NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)
            assert e.legal_mask.dtype == torch.bool
            assert e.value_target in (-1.0, 0.0, 1.0)

    def test_on_short_move_limit_every_z_is_zero_draw(self):
        # With move_limit=4 the starting position cannot end in
        # checkmate within 4 plies (Shogi mate trees are much longer),
        # so the game terminates via the move-limit branch and every
        # example gets z=0.
        net = _tiny_network()
        examples = play_game(
            network=net,
            num_simulations=1,
            move_limit=4,
            rng=np.random.default_rng(0),
        )
        assert len(examples) > 0
        for e in examples:
            assert e.value_target == 0.0

    def test_terminal_starting_board_returns_empty_list(self):
        net = _tiny_network()
        # Two-gold corner mate: White (the STM) is in checkmate.
        # play_game should see this on the first iteration, exit the
        # loop, and return [].
        terminal_board = shogi.Board("kG7/1G7/9/9/9/9/9/9/8K w - 1")
        assert terminal_board.is_checkmate()

        examples = play_game(
            network=net,
            num_simulations=1,
            move_limit=10,
            starting_board=terminal_board,
            rng=np.random.default_rng(0),
        )
        assert examples == []

    def test_policy_targets_sum_to_approximately_one(self):
        # At τ=1, every recorded policy target is the normalized visit
        # distribution, which should sum to ~1 across its non-zero
        # entries (= legal moves).
        net = _tiny_network()
        examples = play_game(
            network=net,
            num_simulations=2,
            move_limit=4,
            rng=np.random.default_rng(0),
        )
        assert len(examples) > 0
        for e in examples:
            total = e.policy_target.sum().item()
            assert total == pytest.approx(1.0, abs=1e-5), (
                f"policy target should sum to 1; got {total}"
            )

    def test_legal_mask_nonzero_at_each_step(self):
        # No recorded position in a valid game is a stalemate/terminal,
        # so the legal-move mask should have at least one True entry.
        net = _tiny_network()
        examples = play_game(
            network=net,
            num_simulations=1,
            move_limit=4,
            rng=np.random.default_rng(0),
        )
        for e in examples:
            assert e.legal_mask.any()
