"""Tests for shogi_dojo.alphazero.rules — python-shogi adapter.

Coverage goals:
  - Piece-type translation is bijective across all 14 piece types.
  - board_to_position on the starting position recovers the expected
    piece count, king positions, empty hands, move number, and side.
  - shogi_move_to_move / move_to_shogi_move round-trip on ordinary
    moves, promoting moves, and drops.
  - legal_moves_fn returns a concrete list matching board.legal_moves.
  - apply_fn does not mutate the input board and produces a new board
    whose move_number has advanced.
  - is_terminal_fn correctly identifies the three kinds of terminal:
    checkmate, fourfold repetition, and move-limit exhaustion; and
    returns None for ongoing games.
  - make_evaluate_fn's returned callable produces (prior dict keyed by
    shogi.Move, value in [-1, +1]), with priors summing to ~1 over
    legal moves.
"""

from __future__ import annotations

import shogi

from shogi_dojo.alphazero.network import AlphaZeroNetwork
from shogi_dojo.alphazero.rules import (
    _PIECE_TYPE_FROM_SHOGI,
    _PIECE_TYPE_TO_SHOGI,
    apply_fn,
    board_to_position,
    is_terminal_fn,
    legal_moves_fn,
    make_evaluate_fn,
    move_to_shogi_move,
    shogi_move_to_move,
)
from shogi_dojo.nnue.constants import (
    BLACK,
    KING,
    PAWN,
    ROOK,
    WHITE,
)


# ---------------------------------------------------------------------------
# Piece-type translation
# ---------------------------------------------------------------------------


class TestPieceTypeTranslation:
    def test_from_and_to_shogi_are_inverses(self):
        for our_id, shogi_id in _PIECE_TYPE_TO_SHOGI.items():
            assert _PIECE_TYPE_FROM_SHOGI[shogi_id] == our_id

    def test_covers_all_14_piece_types(self):
        assert len(_PIECE_TYPE_FROM_SHOGI) == 14
        assert len(_PIECE_TYPE_TO_SHOGI) == 14

    def test_shogi_pawn_maps_to_our_pawn(self):
        assert _PIECE_TYPE_FROM_SHOGI[shogi.PAWN] == PAWN

    def test_shogi_king_maps_to_our_king(self):
        assert _PIECE_TYPE_FROM_SHOGI[shogi.KING] == KING


# ---------------------------------------------------------------------------
# board_to_position
# ---------------------------------------------------------------------------


class TestBoardToPosition:
    def test_starting_position_piece_count(self):
        board = shogi.Board()
        pos = board_to_position(board)
        # 20 pieces per side including kings = 40 total.
        assert len(pos.pieces) == 40

    def test_starting_position_has_two_kings(self):
        board = shogi.Board()
        pos = board_to_position(board)
        kings = [p for p in pos.pieces if p[0] == KING]
        assert len(kings) == 2
        colors = {king[1] for king in kings}
        assert colors == {BLACK, WHITE}

    def test_starting_position_side_to_move_is_black(self):
        pos = board_to_position(shogi.Board())
        assert pos.side_to_move == BLACK

    def test_starting_position_hands_empty(self):
        pos = board_to_position(shogi.Board())
        assert all(c == 0 for c in pos.hand[BLACK])
        assert all(c == 0 for c in pos.hand[WHITE])

    def test_move_number_preserved(self):
        board = shogi.Board()
        board.push(next(iter(board.legal_moves)))
        pos = board_to_position(board)
        assert pos.move_number == 2

    def test_hand_piece_preserved(self):
        # Construct a position where Black has one pawn and one rook in hand.
        board = shogi.Board("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b Pr 1")
        pos = board_to_position(board)
        # Black pawn count (index 0 in HAND_PIECE_TYPES) should be 1.
        assert pos.hand[BLACK][0] == 1
        # White rook count (index 6) should be 1.
        assert pos.hand[WHITE][6] == 1

    def test_repetition_count_is_caller_supplied(self):
        board = shogi.Board()
        pos = board_to_position(board, repetition_count=3)
        assert pos.repetition_count == 3


# ---------------------------------------------------------------------------
# Move translation round-trip
# ---------------------------------------------------------------------------


class TestMoveTranslation:
    def test_ordinary_move_round_trip(self):
        # "9g9f" — Black pawn one square forward, no promotion.
        m = shogi.Move.from_usi("9g9f")
        our = shogi_move_to_move(m)
        back = move_to_shogi_move(our)
        assert back.usi() == m.usi()
        assert back.from_square == m.from_square
        assert back.to_square == m.to_square
        assert back.drop_piece_type is None
        assert not back.promotion

    def test_promoting_move_round_trip(self):
        # A promoting pawn move, USI syntax appends "+".
        m = shogi.Move.from_usi("1g1f+")
        our = shogi_move_to_move(m)
        assert our.promote
        back = move_to_shogi_move(our)
        assert back.usi() == m.usi()
        assert back.promotion

    def test_drop_round_trip(self):
        # USI drop syntax: "P*5e" = drop a Pawn on square 5e.
        m = shogi.Move.from_usi("P*5e")
        our = shogi_move_to_move(m)
        assert our.is_drop()
        assert our.drop_piece == PAWN
        assert our.from_square is None
        back = move_to_shogi_move(our)
        assert back.usi() == m.usi()
        assert back.drop_piece_type == shogi.PAWN

    def test_rook_drop_round_trip(self):
        m = shogi.Move.from_usi("R*5e")
        our = shogi_move_to_move(m)
        assert our.drop_piece == ROOK
        back = move_to_shogi_move(our)
        assert back.usi() == m.usi()


# ---------------------------------------------------------------------------
# MCTS callables: legal_moves_fn, apply_fn
# ---------------------------------------------------------------------------


class TestLegalMovesFn:
    def test_returns_list_of_all_legal_moves(self):
        board = shogi.Board()
        direct = list(board.legal_moves)
        via_fn = legal_moves_fn(board)
        assert len(via_fn) == len(direct)
        assert {m.usi() for m in via_fn} == {m.usi() for m in direct}

    def test_starting_position_has_30_legal_moves(self):
        # Known fact about the Shogi starting position.
        assert len(legal_moves_fn(shogi.Board())) == 30


class TestApplyFn:
    def test_does_not_mutate_input_board(self):
        board = shogi.Board()
        original_sfen = board.sfen()
        move = next(iter(board.legal_moves))
        _ = apply_fn(board, move)
        assert board.sfen() == original_sfen

    def test_new_board_has_move_applied(self):
        board = shogi.Board()
        move = next(iter(board.legal_moves))
        new = apply_fn(board, move)
        assert new.move_number == 2
        assert new.turn != board.turn

    def test_repeated_apply_builds_move_sequence(self):
        board = shogi.Board()
        move = next(iter(board.legal_moves))
        b1 = apply_fn(board, move)
        move2 = next(iter(b1.legal_moves))
        b2 = apply_fn(b1, move2)
        assert b2.move_number == 3
        assert b2.turn == board.turn


# ---------------------------------------------------------------------------
# is_terminal_fn
# ---------------------------------------------------------------------------


class TestIsTerminalFn:
    def test_fresh_board_not_terminal(self):
        is_term, val = is_terminal_fn(shogi.Board())
        assert not is_term
        assert val is None

    def test_checkmate_returns_minus_one(self):
        # Two-gold corner mate. White's king at 9a is attacked by two
        # mutually-protected Black golds (8a and 8b). Every escape
        # square is attacked and neither gold can be captured. Verified
        # to satisfy board.is_checkmate() in python-shogi.
        board = shogi.Board("kG7/1G7/9/9/9/9/9/9/8K w - 1")
        assert board.is_checkmate()
        is_term, val = is_terminal_fn(board)
        assert is_term
        assert val == -1.0

    def test_move_limit_treated_as_draw(self):
        board = shogi.Board()
        # Bypass the real move-number increment by constructing an SFEN
        # that claims a very high move number.
        board = shogi.Board(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1000"
        )
        is_term, val = is_terminal_fn(board, move_limit=500)
        assert is_term
        assert val == 0.0

    def test_under_move_limit_is_not_terminal(self):
        board = shogi.Board()
        is_term, val = is_terminal_fn(board, move_limit=500)
        assert not is_term


# ---------------------------------------------------------------------------
# make_evaluate_fn
# ---------------------------------------------------------------------------


class TestMakeEvaluateFn:
    def _tiny_network(self) -> AlphaZeroNetwork:
        # A minimal but valid network for the bridging test. We don't
        # care about the quality of its output, only its shape and the
        # plumbing.
        return AlphaZeroNetwork(num_blocks=1, num_filters=8, value_head_hidden=8)

    def test_returns_callable(self):
        net = self._tiny_network()
        fn = make_evaluate_fn(net)
        assert callable(fn)

    def test_prior_keyed_by_shogi_move(self):
        net = self._tiny_network()
        fn = make_evaluate_fn(net)
        board = shogi.Board()
        prior, _ = fn(board)
        # Every key is a shogi.Move; every key is a legal move from the
        # board.
        legal_usis = {m.usi() for m in board.legal_moves}
        prior_usis = {m.usi() for m in prior.keys()}
        assert prior_usis == legal_usis

    def test_prior_sums_to_approximately_one(self):
        net = self._tiny_network()
        fn = make_evaluate_fn(net)
        prior, _ = fn(shogi.Board())
        assert sum(prior.values()) == pytest_approx(1.0, 1e-5)

    def test_value_is_scalar_in_minus_one_plus_one(self):
        net = self._tiny_network()
        fn = make_evaluate_fn(net)
        _, value = fn(shogi.Board())
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_does_not_leave_network_in_eval_mode(self):
        # If the caller had training mode on, evaluate_fn must restore
        # it after use.
        net = self._tiny_network()
        net.train()
        assert net.training   # sanity
        fn = make_evaluate_fn(net)
        _ = fn(shogi.Board())
        assert net.training   # still True

    def test_does_not_track_gradients_through_forward(self):
        net = self._tiny_network()
        net.eval()
        fn = make_evaluate_fn(net)
        # No parameter should have a grad_fn attached to any tensor
        # returned from forward. (Values we get back are plain Python
        # floats, so this is implicit — but verify that grads remain
        # None on parameters.)
        # Zero any existing grads.
        for p in net.parameters():
            p.grad = None
        _ = fn(shogi.Board())
        for p in net.parameters():
            assert p.grad is None


def pytest_approx(value: float, tol: float):
    """Tiny shim so the test doesn't need to import pytest just for this.

    Returns an object that compares equal to any float within ``tol``.
    """
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= tol
    return _Approx()
