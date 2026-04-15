"""AlphaZero-style board and move encoding for Shogi.

This module converts a Shogi position into the dense input planes expected
by an AlphaZero-style policy/value network, and provides the bidirectional
mapping between Shogi moves and the flat policy index space used by the
network's policy head.

Two forms of encoding live here:

  1. Position encoding (:func:`encode_position`): maps a board state to a
     dense float tensor of shape ``(C, NUM_RANKS, NUM_FILES)`` where
     ``C = NUM_INPUT_CHANNELS = 45``. We use a *simplified, single-position*
     version of the AlphaZero paper's encoding — no history planes, no
     repetition planes beyond a single normalized scalar — to keep this
     research-survey implementation readable. The full paper-faithful
     119-plane scheme (45 per-position planes × 8-ply history + a few
     constants) is described in
     ``doc/alphazero/AlphaZero_Reference.md`` §5; extending this module
     to match it later is straightforward.

  2. Move encoding (:func:`encode_move` / :func:`decode_move` /
     :func:`legal_move_mask`): maps a Shogi :class:`Move` to a flat policy
     index in ``[0, POLICY_DIM)``, where ``POLICY_DIM = 11_259 = 9 × 9 × 139``.
     This *is* paper-faithful — we deliberately use the full 9×9×139 action
     layout from the AlphaZero paper's appendix so that students can read
     the original alongside this code without translation.

Side-to-move convention:

  All encodings produce tensors *from the side-to-move's perspective*. When
  it is White's turn, the board is conceptually rotated 180° before
  encoding (so the network always sees its own pieces at the bottom of the
  input planes), and moves are flipped to match. This matches the
  convention used in the NNUE pipeline; see
  ``doc/nnue/Model_Training_Reference.md`` §1 for the parallel discussion.

This module imports a small number of *shared Shogi facts* (piece type
IDs, board dimensions, hand piece list) from
:mod:`shogi_dojo.nnue.constants`. These are not NNUE-specific concepts;
they are Shogi-game facts that happen to live in the ``nnue/`` tree
today. If a third paradigm is added, factoring them into a shared module
would be the next step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from shogi_dojo.nnue.constants import (
    BISHOP,
    BLACK,
    DRAGON,
    GOLD,
    HAND_PIECE_MAX,
    HAND_PIECE_TYPES,
    HORSE,
    KING,
    KNIGHT,
    LANCE,
    NUM_FILES,
    NUM_RANKS,
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
# Position encoding constants
# ============================================================================
#
# We encode a single position (no history) as a stack of binary or
# normalized planes of shape (NUM_RANKS, NUM_FILES). Each plane describes
# one feature; the whole stack has shape (NUM_INPUT_CHANNELS, 9, 9).
#
# Plane layout (45 channels total):
#
#   Channels  0..13   Own pieces, one plane per piece type (P, L, N, S, G,
#                     B, R, K, +P, +L, +N, +S, +B, +R)
#                     Plane[i, y, x] = 1.0 if own piece of type i on (y, x)
#
#   Channels 14..27   Opponent pieces, one plane per piece type (same order)
#                     Plane[14+i, y, x] = 1.0 if opponent piece of type i
#                                              on (y, x)
#
#   Channels 28..34   Own hand piece counts, one plane per droppable type
#                     (P, L, N, S, G, B, R). The plane is filled with the
#                     normalized count (count / max_count), broadcast
#                     across all (y, x).
#
#   Channels 35..41   Opponent hand piece counts (same scheme)
#
#   Channel  42       Repetition count (normalized). The plane is filled
#                     with min(rep_count, 3) / 3.0 to give the network a
#                     bounded repetition signal.
#
#   Channel  43       Side-to-move plane (after perspective flip, this is
#                     always 1.0 — STM is always "the player at the
#                     bottom"). Kept as a plane for didactic clarity; the
#                     network can ignore it if it learns the convention.
#
#   Channel  44       Move number (normalized). The plane is filled with
#                     min(move_number, 200) / 200.0 to give the network a
#                     bounded sense of how late in the game we are.
#
# This is a deliberate simplification of the paper's 119-plane scheme.
# History planes (the paper stacks the last 8 plies' encodings) are
# omitted: encoding history requires the caller to maintain a Position
# trail across moves, which is out of scope for a single per-position
# encoder. Adding history later would extend this module into a stateful
# encoder class.

# Piece-type ordering used for the per-piece planes. Matches the existing
# nnue/constants.py ordering, with KING tacked onto the end so that all
# 14 piece types (including King) have a plane.
ALL_PIECE_TYPES_ORDERED = [
    PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK,
    KING,
    PRO_PAWN, PRO_LANCE, PRO_KNIGHT, PRO_SILVER, HORSE, DRAGON,
]
NUM_ALL_PIECE_TYPES = len(ALL_PIECE_TYPES_ORDERED)  # 14

# Plane offsets within the input tensor.
_OWN_PIECE_PLANE_BASE = 0
_OPP_PIECE_PLANE_BASE = NUM_ALL_PIECE_TYPES                          # 14
_OWN_HAND_PLANE_BASE = 2 * NUM_ALL_PIECE_TYPES                       # 28
_OPP_HAND_PLANE_BASE = _OWN_HAND_PLANE_BASE + len(HAND_PIECE_TYPES)  # 35
_REP_PLANE = _OPP_HAND_PLANE_BASE + len(HAND_PIECE_TYPES)            # 42
_STM_PLANE = _REP_PLANE + 1                                          # 43
_PLY_PLANE = _STM_PLANE + 1                                          # 44

NUM_INPUT_CHANNELS = _PLY_PLANE + 1                                  # 45

# Caps for the normalized scalar planes. Chosen as round numbers that
# comfortably exceed typical Shogi values; the exact cap matters less than
# having a fixed, documented divisor so that two callers agree on scale.
REPETITION_CAP = 3        # Sennichite kicks in at the 4th occurrence
PLY_CAP = 200             # Reasonable upper bound for typical games


# ============================================================================
# Move encoding constants
# ============================================================================
#
# Per the AlphaZero paper's Shogi encoding (arXiv:1712.01815, Appendix A,
# Table S3), the action space at every square is decomposed into 139
# planes, organized as:
#
#   Planes 0..63     Sliding "queen" moves: 8 directions × 8 distances.
#                     Plane = direction_index * 8 + (distance - 1).
#   Planes 64..65    Knight moves (Shogi knight only goes forward two ranks
#                     ± one file with a jump). Plane 64 = forward-left,
#                     plane 65 = forward-right.
#   Planes 66..129   Promoting variants of planes 0..63 (board move with
#                     promotion).
#   Planes 130..131  Promoting variants of planes 64..65.
#   Planes 132..138  Drops, one plane per droppable piece type, indexed at
#                     the destination square (drops have no source square).
#
# Total per square: 64 + 2 + 64 + 2 + 7 = 139 planes.
# Total action space: 9 × 9 × 139 = 11,259 = POLICY_DIM.

# Direction ordering for the 8 queen-like sliding directions, expressed as
# (delta_y, delta_x) from a "forward = decreasing y" perspective. The
# perspective is always the side-to-move's, so "north" / "forward" is
# always toward the opponent.
_QUEEN_DIRECTIONS = [
    (-1,  0),   # 0 N  (forward)
    (-1, +1),   # 1 NE
    ( 0, +1),   # 2 E
    (+1, +1),   # 3 SE
    (+1,  0),   # 4 S  (backward)
    (+1, -1),   # 5 SW
    ( 0, -1),   # 6 W
    (-1, -1),   # 7 NW
]
NUM_QUEEN_DIRECTIONS = len(_QUEEN_DIRECTIONS)  # 8
NUM_QUEEN_DISTANCES = 8                         # 1..8 inclusive
NUM_QUEEN_PLANES = NUM_QUEEN_DIRECTIONS * NUM_QUEEN_DISTANCES  # 64

# Knight offsets, in the same forward-is-negative-y convention. The Shogi
# knight is unusual: it only moves forward, jumps over intervening pieces,
# and lands two ranks forward, one file to either side.
_KNIGHT_OFFSETS = [
    (-2, -1),   # 0  knight forward-left
    (-2, +1),   # 1  knight forward-right
]
NUM_KNIGHT_OFFSETS = len(_KNIGHT_OFFSETS)  # 2

# Plane-index ranges within the 139-plane stack.
NUM_SLIDING_PLANES = NUM_QUEEN_PLANES + NUM_KNIGHT_OFFSETS  # 66
NUM_PROMOTING_PLANES = NUM_SLIDING_PLANES                    # 66
NUM_DROP_PLANES = len(HAND_PIECE_TYPES)                      # 7

NUM_MOVE_PLANES = NUM_SLIDING_PLANES + NUM_PROMOTING_PLANES + NUM_DROP_PLANES  # 139

# Plane offsets — start of each subgroup within the 139-plane stack.
_QUEEN_PLANE_BASE = 0                                          # 0
_KNIGHT_PLANE_BASE = NUM_QUEEN_PLANES                          # 64
_PROMOTING_QUEEN_PLANE_BASE = NUM_SLIDING_PLANES               # 66
_PROMOTING_KNIGHT_PLANE_BASE = _PROMOTING_QUEEN_PLANE_BASE + NUM_QUEEN_PLANES  # 130
_DROP_PLANE_BASE = NUM_SLIDING_PLANES + NUM_PROMOTING_PLANES   # 132

POLICY_DIM = NUM_MOVE_PLANES * NUM_SQUARES   # 9 × 9 × 139 = 11,259


# ============================================================================
# Public data types
# ============================================================================


@dataclass(frozen=True)
class Position:
    """A complete Shogi position from which an input tensor can be built.

    This is intentionally a simple, framework-independent data class. It can
    be produced from an SFEN string (see :func:`position_from_sfen`) or
    constructed manually for tests.

    Attributes:
        pieces: List of ``(piece_type, color, square)`` tuples, including
            kings. ``piece_type`` uses the IDs from
            :mod:`shogi_dojo.nnue.constants` (``PAWN``..``DRAGON``,
            ``KING``). ``square`` is in 0..80 row-major board coordinates,
            *not* perspective-flipped.
        side_to_move: ``BLACK`` (0) or ``WHITE`` (1). Determines whose
            perspective the encoder uses.
        hand: ``hand[color][i]`` = how many of ``HAND_PIECE_TYPES[i]`` that
            color holds. Same shape as the hand returned by
            ``shogi_dojo.nnue.features.parse_sfen``.
        move_number: Full move counter (1-indexed) from the SFEN.
        repetition_count: How many times the *current* position has
            previously occurred in the game. AlphaZero uses this as part
            of the input so the network can avoid sennichite. Defaults
            to 0 if the caller is not tracking repetitions.
    """

    pieces: tuple[tuple[int, int, int], ...]
    side_to_move: int
    hand: tuple[tuple[int, ...], tuple[int, ...]]
    move_number: int = 1
    repetition_count: int = 0


@dataclass(frozen=True)
class Move:
    """A Shogi move expressed in raw board coordinates.

    Two flavors:

    - **Board moves**: ``from_square`` is the source square (0..80),
      ``to_square`` is the destination, ``drop_piece`` is ``None``.
      ``promote`` is ``True`` if the move includes promotion.

    - **Drops**: ``from_square`` is ``None``, ``to_square`` is the
      destination, ``drop_piece`` is one of the droppable piece type
      IDs (``PAWN``..``ROOK``). ``promote`` must be ``False`` (drops
      cannot promote).

    Coordinates are in the canonical board frame (top-left = 0,
    bottom-right = 80). The encoder applies any perspective flip
    internally based on the ``side_to_move`` argument.
    """

    from_square: int | None
    to_square: int
    drop_piece: int | None = None
    promote: bool = False

    def is_drop(self) -> bool:
        return self.from_square is None


# ============================================================================
# Coordinate helpers
# ============================================================================


def _square_to_yx(square: int) -> tuple[int, int]:
    """Decompose a 0..80 square index into ``(rank, file)`` pair."""
    return square // NUM_FILES, square % NUM_FILES


def _yx_to_square(y: int, x: int) -> int:
    """Compose a ``(rank, file)`` pair into a 0..80 square index."""
    return y * NUM_FILES + x


def _flip_yx(y: int, x: int) -> tuple[int, int]:
    """Rotate a square 180° (used for White's perspective)."""
    return NUM_RANKS - 1 - y, NUM_FILES - 1 - x


def _flip_square(square: int) -> int:
    """Rotate a square index 180°."""
    y, x = _square_to_yx(square)
    fy, fx = _flip_yx(y, x)
    return _yx_to_square(fy, fx)


# ============================================================================
# Position → input tensor
# ============================================================================


def encode_position(pos: Position) -> torch.Tensor:
    """Convert a Shogi :class:`Position` into a dense input tensor.

    The output is always presented from the side-to-move's perspective:
    when ``pos.side_to_move == WHITE``, the board is flipped 180° and the
    piece colors are swapped before being written into the planes. This
    means the network only ever has to learn one orientation.

    Args:
        pos: The position to encode.

    Returns:
        A float tensor of shape ``(NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)``,
        i.e. ``(45, 9, 9)``.
    """
    tensor = torch.zeros(NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES, dtype=torch.float32)

    flip = pos.side_to_move == WHITE

    # Build a piece-type → plane-offset lookup so that the per-piece loop
    # below is a clean dict access rather than a long if/elif ladder.
    piece_type_to_plane = {pt: i for i, pt in enumerate(ALL_PIECE_TYPES_ORDERED)}

    # --- Per-piece planes ---
    for piece_type, color, square in pos.pieces:
        plane_offset = piece_type_to_plane[piece_type]

        # Apply perspective flip if STM is White:
        #   - Square is rotated 180°
        #   - Color is swapped (own ↔ opponent)
        if flip:
            y, x = _flip_yx(*_square_to_yx(square))
            owner_is_stm = color == WHITE
        else:
            y, x = _square_to_yx(square)
            owner_is_stm = color == BLACK

        plane = (
            _OWN_PIECE_PLANE_BASE + plane_offset
            if owner_is_stm
            else _OPP_PIECE_PLANE_BASE + plane_offset
        )
        tensor[plane, y, x] = 1.0

    # --- Hand piece planes ---
    # Hand counts are broadcast across the whole 9×9 plane, scaled to [0, 1]
    # by dividing by the maximum possible count of that piece type.
    own_color = WHITE if flip else BLACK
    opp_color = 1 - own_color

    for i, pt in enumerate(HAND_PIECE_TYPES):
        max_count = HAND_PIECE_MAX[pt]
        own_norm = pos.hand[own_color][i] / max_count
        opp_norm = pos.hand[opp_color][i] / max_count
        tensor[_OWN_HAND_PLANE_BASE + i].fill_(own_norm)
        tensor[_OPP_HAND_PLANE_BASE + i].fill_(opp_norm)

    # --- Repetition plane ---
    rep_norm = min(pos.repetition_count, REPETITION_CAP) / REPETITION_CAP
    tensor[_REP_PLANE].fill_(rep_norm)

    # --- Side-to-move plane ---
    # After perspective flipping, "STM is at the bottom" is true by
    # construction, so this plane is always 1.0. We keep it for didactic
    # parity with the AlphaZero paper, where the constant STM plane is
    # documented as part of the encoding.
    tensor[_STM_PLANE].fill_(1.0)

    # --- Move number plane ---
    ply_norm = min(pos.move_number, PLY_CAP) / PLY_CAP
    tensor[_PLY_PLANE].fill_(ply_norm)

    return tensor


# ============================================================================
# Move ↔ policy index
# ============================================================================
#
# A flat policy index in [0, POLICY_DIM) decomposes as:
#
#     plane = index // NUM_SQUARES
#     y     = (index % NUM_SQUARES) // NUM_FILES
#     x     = (index % NUM_SQUARES) % NUM_FILES
#
# For board moves the (y, x) coordinates are the *source* square (after
# perspective flip), and the plane encodes the move geometry.
# For drops the (y, x) coordinates are the *destination* square (after
# perspective flip), and the plane encodes which piece is being dropped.


def encode_move(move: Move, side_to_move: int) -> int:
    """Encode a Shogi move into a flat policy index in ``[0, POLICY_DIM)``.

    The move is given in canonical board coordinates; the encoder applies
    a 180° perspective flip when ``side_to_move == WHITE`` so the resulting
    index is always relative to "the side-to-move sits at the bottom of
    the board, moving forward = decreasing rank."

    Args:
        move: The move to encode.
        side_to_move: ``BLACK`` or ``WHITE``.

    Returns:
        A policy index in ``[0, POLICY_DIM)``.

    Raises:
        ValueError: if the move's geometry is not representable in the
            9×9×139 action space (e.g. a sliding distance > 8 squares,
            which cannot occur on a 9×9 board, or an unknown drop piece).
    """
    flip = side_to_move == WHITE

    if move.is_drop():
        # --- Drop ---
        if move.drop_piece is None:
            raise ValueError("drop move must have drop_piece set")
        if move.drop_piece not in HAND_PIECE_TYPES:
            raise ValueError(
                f"drop_piece must be one of HAND_PIECE_TYPES; got {move.drop_piece}"
            )
        if move.promote:
            raise ValueError("drops cannot promote")

        to_y, to_x = _square_to_yx(move.to_square)
        if flip:
            to_y, to_x = _flip_yx(to_y, to_x)

        drop_plane_offset = HAND_PIECE_TYPES.index(move.drop_piece)
        plane = _DROP_PLANE_BASE + drop_plane_offset
        return _index_for(plane, to_y, to_x)

    # --- Board move ---
    if move.from_square is None:
        raise ValueError("board move must have from_square set")

    from_y, from_x = _square_to_yx(move.from_square)
    to_y, to_x = _square_to_yx(move.to_square)
    if flip:
        from_y, from_x = _flip_yx(from_y, from_x)
        to_y, to_x = _flip_yx(to_y, to_x)

    delta_y = to_y - from_y
    delta_x = to_x - from_x

    # Try the knight-move encoding first, since knight offsets coincide with
    # no queen direction.
    knight_idx = _match_knight_offset(delta_y, delta_x)
    if knight_idx is not None:
        plane_base = _PROMOTING_KNIGHT_PLANE_BASE if move.promote else _KNIGHT_PLANE_BASE
        plane = plane_base + knight_idx
        return _index_for(plane, from_y, from_x)

    # Otherwise it must be a queen-like sliding move: same direction
    # repeated 1..8 times.
    queen = _match_queen_move(delta_y, delta_x)
    if queen is None:
        raise ValueError(
            f"move from {move.from_square} to {move.to_square} (delta=({delta_y},{delta_x})) "
            f"is not representable as a queen-like or knight move"
        )
    direction_idx, distance = queen
    plane_base = _PROMOTING_QUEEN_PLANE_BASE if move.promote else _QUEEN_PLANE_BASE
    plane = plane_base + direction_idx * NUM_QUEEN_DISTANCES + (distance - 1)
    return _index_for(plane, from_y, from_x)


def decode_move(index: int, side_to_move: int) -> Move:
    """Inverse of :func:`encode_move`.

    Given a flat policy index and the side-to-move, reconstruct the
    :class:`Move` in canonical board coordinates.

    This function does *not* validate that the resulting move is legal
    (or even possible — e.g. a queen plane can encode a move that walks
    off the board). It only performs the geometric decode. Callers should
    use :func:`legal_move_mask` to filter out impossible/illegal moves
    before sampling from the policy.

    Args:
        index: A policy index in ``[0, POLICY_DIM)``.
        side_to_move: ``BLACK`` or ``WHITE``.

    Returns:
        The decoded :class:`Move` in canonical (unflipped) coordinates.

    Raises:
        ValueError: if ``index`` is outside ``[0, POLICY_DIM)``, or if the
            decoded geometry walks off the 9×9 board.
    """
    if not 0 <= index < POLICY_DIM:
        raise ValueError(f"index {index} outside [0, {POLICY_DIM})")
    flip = side_to_move == WHITE

    plane = index // NUM_SQUARES
    sq = index % NUM_SQUARES
    y, x = _square_to_yx(sq)

    if plane >= _DROP_PLANE_BASE:
        # --- Drop ---
        drop_plane_offset = plane - _DROP_PLANE_BASE
        drop_piece = HAND_PIECE_TYPES[drop_plane_offset]
        if flip:
            y, x = _flip_yx(y, x)
        return Move(
            from_square=None,
            to_square=_yx_to_square(y, x),
            drop_piece=drop_piece,
            promote=False,
        )

    # --- Board move (queen or knight, possibly promoting) ---
    promote = plane >= _PROMOTING_QUEEN_PLANE_BASE
    geom_plane = plane - _PROMOTING_QUEEN_PLANE_BASE if promote else plane

    if geom_plane >= _KNIGHT_PLANE_BASE:
        # Knight
        knight_idx = geom_plane - _KNIGHT_PLANE_BASE
        delta_y, delta_x = _KNIGHT_OFFSETS[knight_idx]
    else:
        # Queen-like
        direction_idx = geom_plane // NUM_QUEEN_DISTANCES
        distance = (geom_plane % NUM_QUEEN_DISTANCES) + 1
        dy, dx = _QUEEN_DIRECTIONS[direction_idx]
        delta_y, delta_x = dy * distance, dx * distance

    # `(y, x)` is the from-square in the (possibly flipped) STM frame.
    to_y, to_x = y + delta_y, x + delta_x
    if not (0 <= to_y < NUM_RANKS and 0 <= to_x < NUM_FILES):
        raise ValueError(
            f"decoded move from ({y},{x}) by ({delta_y},{delta_x}) walks off the board"
        )

    # Convert back to canonical coordinates (undo perspective flip).
    if flip:
        from_y, from_x = _flip_yx(y, x)
        to_y, to_x = _flip_yx(to_y, to_x)
    else:
        from_y, from_x = y, x

    return Move(
        from_square=_yx_to_square(from_y, from_x),
        to_square=_yx_to_square(to_y, to_x),
        drop_piece=None,
        promote=promote,
    )


def legal_move_mask(legal_moves: list[Move], side_to_move: int) -> torch.Tensor:
    """Build a boolean mask over the 9×9×139 action space.

    The mask is ``True`` exactly at the policy indices that correspond to
    moves in ``legal_moves``. Use it to mask the policy logits before
    softmax (set illegal positions to a large negative number) so that
    illegal actions receive zero probability.

    Args:
        legal_moves: The legal moves at the current position.
        side_to_move: ``BLACK`` or ``WHITE``.

    Returns:
        A boolean tensor of shape ``(NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)``,
        i.e. ``(139, 9, 9)``.
    """
    mask = torch.zeros(NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES, dtype=torch.bool)
    for move in legal_moves:
        idx = encode_move(move, side_to_move)
        plane = idx // NUM_SQUARES
        sq = idx % NUM_SQUARES
        y, x = _square_to_yx(sq)
        mask[plane, y, x] = True
    return mask


# ============================================================================
# Internal helpers
# ============================================================================


def _index_for(plane: int, y: int, x: int) -> int:
    """Compose a flat policy index from ``(plane, y, x)``."""
    return plane * NUM_SQUARES + _yx_to_square(y, x)


def _match_knight_offset(delta_y: int, delta_x: int) -> int | None:
    """If ``(delta_y, delta_x)`` matches a knight offset, return its index.

    Knight offsets are ``(-2, ±1)`` in the STM-perspective frame (forward
    = decreasing y). Returns ``None`` if the delta is not a knight move.
    """
    for i, (kdy, kdx) in enumerate(_KNIGHT_OFFSETS):
        if delta_y == kdy and delta_x == kdx:
            return i
    return None


def _match_queen_move(delta_y: int, delta_x: int) -> tuple[int, int] | None:
    """If ``(delta_y, delta_x)`` is a queen-like move, return ``(direction_idx, distance)``.

    A queen-like move travels in one of the eight compass directions for
    1..8 squares. ``(0, 0)`` is rejected (a non-move), as are diagonals
    that don't satisfy ``|delta_y| == |delta_x|`` and orthogonals that
    aren't pure horizontal or vertical.

    Returns ``None`` if the delta does not match any queen-like geometry.
    """
    if delta_y == 0 and delta_x == 0:
        return None

    # The distance is the maximum absolute coordinate change. For a valid
    # queen move, both axes must agree on the distance (diagonal) or one
    # axis must be zero (orthogonal).
    if delta_x == 0:
        distance = abs(delta_y)
        unit_y = 1 if delta_y > 0 else -1
        unit_x = 0
    elif delta_y == 0:
        distance = abs(delta_x)
        unit_y = 0
        unit_x = 1 if delta_x > 0 else -1
    elif abs(delta_y) == abs(delta_x):
        distance = abs(delta_y)
        unit_y = 1 if delta_y > 0 else -1
        unit_x = 1 if delta_x > 0 else -1
    else:
        return None

    if not 1 <= distance <= NUM_QUEEN_DISTANCES:
        return None

    for i, (dy, dx) in enumerate(_QUEEN_DIRECTIONS):
        if dy == unit_y and dx == unit_x:
            return i, distance
    return None


# ============================================================================
# SFEN convenience wrapper
# ============================================================================


def position_from_sfen(sfen: str) -> Position:
    """Convenience wrapper that parses an SFEN string into a :class:`Position`.

    Reuses the SFEN parser from :mod:`shogi_dojo.nnue.features`, which is
    a Shogi-game fact rather than an NNUE-specific concept. The parser
    omits kings from its piece list (NNUE handles them separately via
    king buckets); we reattach them as plain pieces with ``KING`` type
    so that the AlphaZero encoder can include them in the per-piece
    planes.

    Args:
        sfen: SFEN position string (e.g. the standard starting position).

    Returns:
        A :class:`Position` with ``repetition_count=0``. Callers tracking
        repetitions across moves can construct positions directly with a
        non-zero count.
    """
    # Local import to avoid a hard module-load-time dependency on the
    # NNUE parser; encoding.py is otherwise NNUE-free.
    from shogi_dojo.nnue.features import parse_sfen

    pieces, side_to_move, hand, black_king_sq, white_king_sq = parse_sfen(sfen)

    pieces_with_kings: list[tuple[int, int, int]] = list(pieces)
    pieces_with_kings.append((KING, BLACK, black_king_sq))
    pieces_with_kings.append((KING, WHITE, white_king_sq))

    # Move number is the 4th SFEN field if present; default to 1.
    parts = sfen.split()
    move_number = int(parts[3]) if len(parts) >= 4 and parts[3].isdigit() else 1

    return Position(
        pieces=tuple(pieces_with_kings),
        side_to_move=side_to_move,
        hand=(tuple(hand[BLACK]), tuple(hand[WHITE])),
        move_number=move_number,
        repetition_count=0,
    )
