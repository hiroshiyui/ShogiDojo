"""python-shogi adapter: the four MCTS callables for real Shogi.

This module is the bridge between :mod:`shogi_dojo.alphazero.mcts` — a
deliberately game-agnostic search algorithm — and actual Shogi rules
supplied by the `python-shogi <https://pypi.org/project/python-shogi/>`_
package.

The MCTS search takes four callables ``(evaluate_fn, legal_moves_fn,
apply_fn, is_terminal_fn)`` whose ``state`` type is opaque to the
algorithm. We use ``shogi.Board`` directly as that state (rather than
our :class:`shogi_dojo.alphazero.encoding.Position` dataclass), for
three reasons:

1. ``shogi.Board`` already knows how to generate legal moves, apply
   them, and detect terminal conditions.
2. Converting to :class:`Position` eagerly at every tree node would be
   wasteful — we only need the tensor form at evaluation time.
3. ``shogi.Board.push`` / ``pop`` is destructive; we pay a modest
   SFEN-round-trip cost in ``apply_fn`` to maintain pure-functional
   semantics (MCTS expects ``apply_fn`` not to mutate its input).

Conventions confirmed to match between python-shogi and this package:

- Square indexing is 0..80, row-major from the top-left. Square 0 is
  "9a" (file 9, rank a = White's back rank on the left from Black's
  view); square 80 is "1i" (file 1, rank i = Black's back rank on the
  right).
- Colors: ``shogi.BLACK == 0``, ``shogi.WHITE == 1``, matching
  :mod:`shogi_dojo.nnue.constants`.
- Move number starts at 1.

The one convention that *does not* match is piece type IDs:

- ``shogi.PAWN == 1``, ``shogi.LANCE == 2``, ... ``shogi.KING == 8``,
  ``shogi.PROM_PAWN == 9`` ... ``shogi.PROM_ROOK == 14``.
- Our :mod:`shogi_dojo.nnue.constants` starts at 0 and places promoted
  pieces at 7..12, with KING at 13.

The ``_PIECE_TYPE_FROM_SHOGI`` / ``_PIECE_TYPE_TO_SHOGI`` dicts below
translate in both directions.
"""

from __future__ import annotations

from typing import Callable, Optional

import shogi
import torch

from shogi_dojo.alphazero.encoding import (
    Move,
    Position,
    encode_move,
    encode_position,
    legal_move_mask,
)
from shogi_dojo.alphazero.network import masked_log_softmax
from shogi_dojo.nnue.constants import (
    BISHOP,
    BLACK,
    DRAGON,
    GOLD,
    HAND_PIECE_TYPES,
    HORSE,
    KING,
    KNIGHT,
    LANCE,
    NUM_FILES,
    NUM_SQUARES,
    PAWN,
    PRO_KNIGHT,
    PRO_LANCE,
    PRO_PAWN,
    PRO_SILVER,
    ROOK,
    SILVER,
    WHITE,
)


# ============================================================================
# Piece-type translation
# ============================================================================

_PIECE_TYPE_FROM_SHOGI: dict[int, int] = {
    shogi.PAWN: PAWN,
    shogi.LANCE: LANCE,
    shogi.KNIGHT: KNIGHT,
    shogi.SILVER: SILVER,
    shogi.GOLD: GOLD,
    shogi.BISHOP: BISHOP,
    shogi.ROOK: ROOK,
    shogi.KING: KING,
    shogi.PROM_PAWN: PRO_PAWN,
    shogi.PROM_LANCE: PRO_LANCE,
    shogi.PROM_KNIGHT: PRO_KNIGHT,
    shogi.PROM_SILVER: PRO_SILVER,
    shogi.PROM_BISHOP: HORSE,
    shogi.PROM_ROOK: DRAGON,
}
_PIECE_TYPE_TO_SHOGI: dict[int, int] = {v: k for k, v in _PIECE_TYPE_FROM_SHOGI.items()}


# ============================================================================
# Board / Move translation
# ============================================================================


def board_to_position(board: shogi.Board, repetition_count: int = 0) -> Position:
    """Convert a :class:`shogi.Board` into our :class:`Position`.

    The conversion is straightforward because the square indexing and
    color conventions match; only piece type IDs need translation.

    Args:
        board: A python-shogi board.
        repetition_count: How many times the current position has
            occurred earlier in the game. The caller is responsible
            for tracking this (python-shogi exposes
            ``board.is_fourfold_repetition()`` but not a repetition
            *count*). Defaults to 0.

    Returns:
        A :class:`Position` suitable for passing to
        :func:`shogi_dojo.alphazero.encoding.encode_position`.
    """
    pieces: list[tuple[int, int, int]] = []
    for sq in range(NUM_SQUARES):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        pt = _PIECE_TYPE_FROM_SHOGI[piece.piece_type]
        color = BLACK if piece.color == shogi.BLACK else WHITE
        pieces.append((pt, color, sq))

    # Hand: python-shogi exposes Counter-like dicts keyed by shogi piece type.
    # Our Position uses a parallel list ordered by HAND_PIECE_TYPES.
    hand_black = tuple(
        board.pieces_in_hand[shogi.BLACK].get(_PIECE_TYPE_TO_SHOGI[pt], 0)
        for pt in HAND_PIECE_TYPES
    )
    hand_white = tuple(
        board.pieces_in_hand[shogi.WHITE].get(_PIECE_TYPE_TO_SHOGI[pt], 0)
        for pt in HAND_PIECE_TYPES
    )

    return Position(
        pieces=tuple(pieces),
        side_to_move=BLACK if board.turn == shogi.BLACK else WHITE,
        hand=(hand_black, hand_white),
        move_number=board.move_number,
        repetition_count=repetition_count,
    )


def shogi_move_to_move(shogi_move: shogi.Move) -> Move:
    """Convert a :class:`shogi.Move` into our :class:`Move`."""
    if shogi_move.drop_piece_type is not None:
        return Move(
            from_square=None,
            to_square=shogi_move.to_square,
            drop_piece=_PIECE_TYPE_FROM_SHOGI[shogi_move.drop_piece_type],
            promote=False,
        )
    return Move(
        from_square=shogi_move.from_square,
        to_square=shogi_move.to_square,
        drop_piece=None,
        promote=bool(shogi_move.promotion),
    )


def move_to_shogi_move(move: Move) -> shogi.Move:
    """Convert our :class:`Move` into a :class:`shogi.Move`."""
    if move.is_drop():
        return shogi.Move(
            from_square=None,
            to_square=move.to_square,
            drop_piece_type=_PIECE_TYPE_TO_SHOGI[move.drop_piece],
        )
    return shogi.Move(
        from_square=move.from_square,
        to_square=move.to_square,
        promotion=move.promote,
    )


# ============================================================================
# MCTS-compatible callables
# ============================================================================


def legal_moves_fn(board: shogi.Board) -> list[shogi.Move]:
    """Enumerate all legal moves at ``board``.

    Returns a plain list (rather than the generator
    ``board.legal_moves`` returns) so that MCTS can index into it
    repeatedly without re-running move generation.
    """
    return list(board.legal_moves)


def apply_fn(board: shogi.Board, move: shogi.Move) -> shogi.Board:
    """Return a new board with ``move`` applied. Does not mutate input.

    python-shogi does not expose a ``.copy()`` method, so we round-trip
    through SFEN. This is ~1–2 μs per call at didactic scale, which is
    negligible compared to a ~10 ms network evaluation.
    """
    new_board = shogi.Board(board.sfen())
    new_board.push(move)
    return new_board


# A ply limit sentinel used by :func:`is_terminal_fn` to cap runaway
# games. AlphaZero (and self-play engines in general) use a limit in
# the low hundreds; the classical 512 is a safe default.
DEFAULT_MOVE_LIMIT = 512


def is_terminal_fn(
    board: shogi.Board,
    move_limit: int = DEFAULT_MOVE_LIMIT,
) -> tuple[bool, Optional[float]]:
    """Decide whether ``board`` is terminal and return the outcome.

    Outcome is expressed from the side-to-move's perspective:

    - ``-1.0`` — the side-to-move has lost (checkmate, stalemate). In
      Shogi, stalemate counts as a loss for the stalemated side, not a
      draw as in chess.
    - ``0.0``  — draw. We treat all fourfold-repetitions (sennichite)
      as draws. Standard Shogi rules make perpetual check by the
      *attacker* a loss for the attacker, but python-shogi does not
      tag that, and distinguishing it is out of scope for a research
      survey.
    - ``+1.0`` — win for the side-to-move. In practice this branch is
      unreachable here: a win is only finalized when the *opponent*
      gets checkmated on their turn, which flips whose STM we are
      looking at.

    Args:
        board: The position to check.
        move_limit: Ply cap. Games past this limit are treated as
            draws. Defaults to 512.

    Returns:
        ``(is_terminal, value)``. When ``is_terminal`` is ``False``,
        ``value`` is ``None``.
    """
    # Fourfold repetition: sennichite → draw for our purposes.
    if board.is_fourfold_repetition():
        return True, 0.0
    # Checkmate or stalemate: STM loses.
    if board.is_checkmate() or board.is_stalemate():
        return True, -1.0
    # Move limit: treat as draw.
    if board.move_number > move_limit:
        return True, 0.0
    return False, None


# ============================================================================
# Network → evaluate_fn bridge
# ============================================================================


def make_evaluate_fn(
    network: torch.nn.Module,
    device: torch.device | str = "cpu",
) -> Callable[[shogi.Board], tuple[dict[shogi.Move, float], float]]:
    """Build an ``evaluate_fn`` that feeds ``board`` through ``network``.

    The returned callable is suitable for passing to
    :class:`shogi_dojo.alphazero.mcts.MCTS`. Each call:

    1. Converts the board to a 45-channel input tensor via
       :func:`encode_position`.
    2. Runs one forward pass through ``network`` (under ``no_grad``).
    3. Applies the legal-move mask and log-softmax to get a probability
       distribution over legal moves.
    4. Returns ``(prior_dict, value)`` where ``prior_dict`` is keyed by
       the original ``shogi.Move`` objects (so the caller can feed them
       back into :func:`apply_fn` directly without re-translating).

    Args:
        network: An :class:`AlphaZeroNetwork` (or any compatible
            module that returns ``(policy_logits, value)`` with shape
            ``(1, 139, 9, 9)`` and ``(1, 1)``).
        device: Where to run inference. Defaults to CPU for the
            survey's laptop-scale runs.

    Returns:
        An evaluate_fn compatible with :class:`MCTS`.
    """
    device = torch.device(device)
    network = network.to(device)

    def evaluate_fn(
        board: shogi.Board,
    ) -> tuple[dict[shogi.Move, float], float]:
        position = board_to_position(board)
        input_tensor = encode_position(position).unsqueeze(0).to(device)

        legal = list(board.legal_moves)
        if not legal:
            # Not expected on a non-terminal board; MCTS should have
            # caught terminality in is_terminal_fn first. Defensive
            # fallback: value 0, empty prior.
            return {}, 0.0

        was_training = network.training
        network.eval()
        try:
            with torch.no_grad():
                policy_logits, value = network(input_tensor)
        finally:
            network.train(was_training)

        # Translate once so we can both build the mask and key the
        # returned prior by the original shogi.Move objects.
        our_moves = [shogi_move_to_move(m) for m in legal]
        mask = legal_move_mask(our_moves, position.side_to_move).unsqueeze(0).to(device)

        log_probs = masked_log_softmax(policy_logits, mask)
        probs = log_probs.exp().squeeze(0).cpu()   # (139, 9, 9)

        prior: dict[shogi.Move, float] = {}
        for shogi_m, our_m in zip(legal, our_moves):
            idx = encode_move(our_m, position.side_to_move)
            plane = idx // NUM_SQUARES
            sq = idx % NUM_SQUARES
            y = sq // NUM_FILES
            x = sq % NUM_FILES
            prior[shogi_m] = float(probs[plane, y, x])

        return prior, float(value.item())

    return evaluate_fn
