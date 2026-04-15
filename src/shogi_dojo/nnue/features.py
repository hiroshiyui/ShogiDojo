"""SFEN parsing and HalfKAv2 feature extraction for Shogi NNUE.

This module converts a Shogi position (in SFEN text format) into sparse
feature indices that the NNUE neural network can consume.

=== Why Sparse Features? ===

A Shogi board has at most ~40 pieces on it at any time, but the total
feature space is ~20,000 dimensions. So the input is extremely sparse
(only ~40 out of ~20,000 features are "active" / non-zero). Instead of
passing a huge mostly-zero vector, we just pass the list of active indices.
This is much faster and more memory-efficient.

=== Two Perspectives ===

NNUE evaluates the position from TWO perspectives simultaneously:
  1. Side-to-move (STM): the player whose turn it is
  2. Not-side-to-move (NSTM): the opponent

Each perspective sees the board "from its own side". For White's
perspective, the board is flipped vertically (rotated 180 degrees)
and colors are swapped, so the network always sees the position as if
"I am at the bottom of the board, my pieces are uppercase."

The two perspective outputs are concatenated before being fed into the
main evaluation network. This lets the network learn asymmetric patterns
like "having the initiative" vs "defending".
"""

from __future__ import annotations


from .constants import (
    BLACK,
    BOARD_FEATURE_DIM,
    HALF_FEATURE_DIM,
    HAND_FEATURES_PER_SIDE,
    HAND_PIECE_MAX,
    HAND_PIECE_TYPES,
    NUM_COLORS,
    NUM_FILES,
    NUM_RANKS,
    NUM_SQUARES,
    SFEN_PIECE_MAP,
    SFEN_PROMOTED_MAP,
    WHITE,
    king_square_to_bucket,
    mirror_file,
)


def parse_sfen(sfen: str) -> tuple[list[tuple[int, int, int]], int, list[list[int]], int, int]:
    """Parse an SFEN string into a structured board representation.

    SFEN (Shogi Forsyth-Edwards Notation) is the standard text format for
    describing a Shogi position. It has 4 space-separated fields:

        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
         ~~~~~~~~~~~~~~~~~~~~~~~~~~board~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ^ ^ ^
                                                                side hand move#

    Board field rules:
      - Rows are separated by '/'
      - Uppercase letters = Black's pieces, lowercase = White's pieces
      - Numbers = consecutive empty squares (e.g., "9" = entire empty row)
      - '+' before a letter = promoted piece (e.g., "+P" = promoted Pawn)

    Hand field rules:
      - '-' means no pieces in hand
      - Otherwise, a sequence like "S2P" means: 1 Silver + 2 Pawns in hand
      - Uppercase = Black's hand, lowercase = White's hand

    === Validation ===

    Malformed SFEN strings raise :class:`ValueError`. The parser enforces
    Shogi-level invariants: both kings must be present, the board must
    describe exactly 81 squares, only board pieces may be promoted, only
    unpromoted non-king pieces may appear in hand, and hand counts must
    not exceed the physical maximum (e.g., 18 Pawns, 2 Rooks).

    Args:
        sfen: The SFEN string (4 space-separated fields).

    Returns:
        A tuple of:
        - pieces: list of (piece_type, color, square) for all non-king pieces
        - side_to_move: BLACK (0) or WHITE (1)
        - hand: 2D list where hand[color][i] = count of HAND_PIECE_TYPES[i] in that player's hand
        - black_king_sq: square index (0-80) where Black's King sits
        - white_king_sq: square index (0-80) where White's King sits

    Raises:
        ValueError: if the SFEN string is malformed or violates a Shogi
            invariant (missing kings, promoted-piece-in-hand, etc.).
    """
    parts = sfen.split()
    if len(parts) < 3:
        raise ValueError(
            f"SFEN must have at least 3 space-separated fields "
            f"(board, side-to-move, hand); got {len(parts)}: {sfen!r}"
        )
    board_str = parts[0]
    side_str = parts[1]
    if side_str not in ('b', 'w'):
        raise ValueError(
            f"side-to-move must be 'b' or 'w', got {side_str!r}"
        )
    side_to_move = BLACK if side_str == 'b' else WHITE
    hand_str = parts[2]

    # --- Parse the board string ---
    pieces: list[tuple[int, int, int]] = []
    black_king_sq = -1
    white_king_sq = -1
    sq = 0          # Current square index, incremented as we scan left-to-right, top-to-bottom
    promoted = False  # True if we just saw a '+' prefix

    for ch in board_str:
        if ch == '/':
            # Row separator — just skip, sq is already at the right position
            continue
        if ch == '+':
            if promoted:
                raise ValueError(f"doubled '+' promotion prefix in board: {board_str!r}")
            # Next piece is promoted
            promoted = True
            continue
        if ch.isdigit():
            # Skip N empty squares
            if promoted:
                raise ValueError(f"'+' must precede a piece, not a digit in board: {board_str!r}")
            sq += int(ch)
            continue

        # It's a piece character
        upper = ch.upper()             # Normalize to uppercase for lookup
        if upper not in SFEN_PIECE_MAP:
            raise ValueError(f"unknown piece character {ch!r} in board: {board_str!r}")
        color = BLACK if ch.isupper() else WHITE  # Case determines color

        if upper == 'K':
            if promoted:
                raise ValueError(f"King cannot be promoted in board: {board_str!r}")
            # Kings are tracked separately (not added to pieces list)
            if color == BLACK:
                if black_king_sq != -1:
                    raise ValueError(f"more than one Black King in board: {board_str!r}")
                black_king_sq = sq
            else:
                if white_king_sq != -1:
                    raise ValueError(f"more than one White King in board: {board_str!r}")
                white_king_sq = sq
        else:
            # Look up the piece type, handling promotion
            if promoted:
                if upper not in SFEN_PROMOTED_MAP:
                    raise ValueError(
                        f"piece {ch!r} cannot be promoted in board: {board_str!r}"
                    )
                piece_type = SFEN_PROMOTED_MAP[upper]
            else:
                piece_type = SFEN_PIECE_MAP[upper]
            pieces.append((piece_type, color, sq))

        promoted = False
        if sq >= NUM_SQUARES:
            raise ValueError(
                f"board describes more than {NUM_SQUARES} squares: {board_str!r}"
            )
        sq += 1

    if sq != NUM_SQUARES:
        raise ValueError(
            f"board describes {sq} squares, expected {NUM_SQUARES}: {board_str!r}"
        )
    if black_king_sq == -1:
        raise ValueError(f"Black King missing from board: {board_str!r}")
    if white_king_sq == -1:
        raise ValueError(f"White King missing from board: {board_str!r}")

    # --- Parse the hand string ---
    # hand[color][i] = how many of HAND_PIECE_TYPES[i] that player holds
    hand = [[0] * len(HAND_PIECE_TYPES) for _ in range(NUM_COLORS)]
    if hand_str != '-':
        count = 0
        for ch in hand_str:
            if ch.isdigit():
                # Multi-digit counts: accumulate digits (e.g., "18" for 18 Pawns)
                count = count * 10 + int(ch)
            elif ch == '+':
                # Promoted pieces revert to base form when captured, so '+' is
                # never valid in the hand field.
                raise ValueError(f"promoted piece not allowed in hand: {hand_str!r}")
            else:
                # A piece character — if no preceding digit, count defaults to 1
                if count == 0:
                    count = 1
                upper = ch.upper()
                if upper == 'K' or upper not in SFEN_PIECE_MAP:
                    raise ValueError(
                        f"invalid piece {ch!r} in hand: {hand_str!r}"
                    )
                piece_type = SFEN_PIECE_MAP[upper]
                if piece_type not in HAND_PIECE_TYPES:
                    raise ValueError(
                        f"piece {ch!r} cannot be in hand: {hand_str!r}"
                    )
                color = BLACK if ch.isupper() else WHITE
                idx = HAND_PIECE_TYPES.index(piece_type)
                max_count = HAND_PIECE_MAX[piece_type]
                if count > max_count:
                    raise ValueError(
                        f"hand count {count} for {ch!r} exceeds maximum "
                        f"{max_count}: {hand_str!r}"
                    )
                if hand[color][idx] != 0:
                    raise ValueError(
                        f"piece {ch!r} listed twice in hand: {hand_str!r}"
                    )
                hand[color][idx] = count
                count = 0
        if count != 0:
            raise ValueError(f"trailing digits without piece in hand: {hand_str!r}")

    return pieces, side_to_move, hand, black_king_sq, white_king_sq


def make_feature_indices(
    pieces: list[tuple[int, int, int]],
    side_to_move: int,
    hand: list[list[int]],
    black_king_sq: int,
    white_king_sq: int,
) -> tuple[list[int], list[int]]:
    """Compute sparse HalfKAv2 feature indices for a position.

    This is the core of the NNUE feature extraction. For each piece on the
    board, we compute a unique index that encodes:
      - Which king bucket the own king is in
      - What type of piece it is
      - What color the piece is
      - Which square the piece is on

    We do this from TWO perspectives:
      1. Black's perspective: the board as Black sees it (no flipping)
      2. White's perspective: the board flipped 180 degrees, colors swapped

    Both perspectives also apply left-right mirroring for symmetry reduction.

    === Index Formula (for board pieces) ===

        index = king_bucket * HALF_FEATURE_DIM
              + piece_type * NUM_COLORS * NUM_SQUARES
              + piece_color * NUM_SQUARES
              + piece_square

    This creates a unique index in [0, FEATURE_TRANSFORMER_INPUT_DIM).

    Args:
        pieces: List of (piece_type, color, square) for non-king pieces.
        side_to_move: BLACK or WHITE.
        hand: hand[color][i] = count of HAND_PIECE_TYPES[i].
        black_king_sq: Black's king square (0-80).
        white_king_sq: White's king square (0-80).

    Returns:
        (stm_indices, nstm_indices): Two lists of active feature indices.
        stm_indices is from the side-to-move's perspective.
    """
    # Determine the king bucket for each perspective.
    # Black's perspective uses Black's king position directly.
    # White's perspective flips the board first (since White "sits" on the other side).
    black_bucket = king_square_to_bucket(mirror_file(black_king_sq))
    white_bucket = king_square_to_bucket(mirror_file(_flip_square(white_king_sq)))

    black_indices: list[int] = []
    white_indices: list[int] = []

    # --- Board piece features ---
    for piece_type, color, sq in pieces:
        # == Black's perspective ==
        # Black sees the board without any flipping.
        # Apply left-right mirror for symmetry.
        b_sq = mirror_file(sq)
        b_color = color
        b_idx = (
            black_bucket * HALF_FEATURE_DIM
            + (piece_type * NUM_COLORS + b_color) * NUM_SQUARES
            + b_sq
        )
        black_indices.append(b_idx)

        # == White's perspective ==
        # White sees the board rotated 180 degrees, and colors are swapped.
        # This makes White's perspective look just like Black's:
        # "my pieces are in front of me, opponent's pieces are across the board."
        w_sq = mirror_file(_flip_square(sq))   # Rotate the square 180 degrees
        w_color = 1 - color                    # Swap the color
        w_idx = (
            white_bucket * HALF_FEATURE_DIM
            + (piece_type * NUM_COLORS + w_color) * NUM_SQUARES
            + w_sq
        )
        white_indices.append(w_idx)

    # --- Hand piece features ---
    # Hand pieces use "thermometer encoding": if you hold 3 Pawns, indices
    # 0, 1, and 2 in the Pawn section are all active. This gives the network
    # a natural sense of "more pieces = more features active".
    hand_base = BOARD_FEATURE_DIM  # Hand features start after board features
    for color in range(NUM_COLORS):
        offset = 0  # Tracks the starting index for each piece type's section
        for i, pt in enumerate(HAND_PIECE_TYPES):
            count = hand[color][i]
            max_count = HAND_PIECE_MAX[pt]
            for c in range(count):
                # Black perspective: color stays the same
                b_color = color
                b_hand_idx = (
                    black_bucket * HALF_FEATURE_DIM
                    + hand_base
                    + b_color * HAND_FEATURES_PER_SIDE
                    + offset + c
                )
                black_indices.append(b_hand_idx)

                # White perspective: colors are swapped
                w_color = 1 - color
                w_hand_idx = (
                    white_bucket * HALF_FEATURE_DIM
                    + hand_base
                    + w_color * HAND_FEATURES_PER_SIDE
                    + offset + c
                )
                white_indices.append(w_hand_idx)
            offset += max_count  # Advance past this piece type's section

    # Return (stm_indices, nstm_indices):
    # The side-to-move's features come first, the opponent's second.
    if side_to_move == BLACK:
        return black_indices, white_indices
    else:
        return white_indices, black_indices


def _flip_square(sq: int) -> int:
    """Flip a square 180 degrees (rotate the board for opponent's perspective).

    This swaps rank 1 ↔ rank 9, file 1 ↔ file 9, etc.
    Used when computing White's perspective: White "sits on the other side",
    so their view of the board is the 180-degree rotation of Black's view.

    Example:
        Square 0  (top-left)     → Square 80 (bottom-right)
        Square 80 (bottom-right) → Square 0  (top-left)
        Square 40 (center)       → Square 40 (center, unchanged)

    Args:
        sq: Square index (0-80).

    Returns:
        The rotated square index.
    """
    rank = sq // NUM_FILES
    file = sq % NUM_FILES
    return (NUM_RANKS - 1 - rank) * NUM_FILES + (NUM_FILES - 1 - file)


def sfen_to_features(sfen: str) -> tuple[list[int], list[int], int]:
    """Convert an SFEN position string to sparse NNUE feature indices.

    This is the main entry point for feature extraction. It combines
    SFEN parsing and HalfKAv2 index computation into a single call.

    Args:
        sfen: SFEN position string (e.g., "lnsgkgsnl/1r5b1/... b - 1").

    Returns:
        stm_indices:  List of active feature indices for the side-to-move's perspective.
        nstm_indices: List of active feature indices for the opponent's perspective.
        side_to_move: BLACK (0) or WHITE (1).
    """
    pieces, stm, hand, bk, wk = parse_sfen(sfen)
    stm_idx, nstm_idx = make_feature_indices(pieces, stm, hand, bk, wk)
    return stm_idx, nstm_idx, stm
