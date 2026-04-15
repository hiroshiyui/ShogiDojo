"""Shogi board and piece constants for NNUE feature extraction.

=== Background: What is NNUE? ===

NNUE (Efficiently Updatable Neural Network) is a neural network architecture
designed for board game evaluation. It was originally created for Shogi by
Yu Nasu in 2018, and later adopted by Stockfish for chess.

The key idea: instead of evaluating the board from scratch every time,
NNUE uses an "accumulator" that can be incrementally updated when a single
piece moves — making it extremely fast during game-tree search.

=== Background: What is Shogi? ===

Shogi (Japanese Chess) is played on a 9x9 board. Key differences from chess:
  - Captured pieces can be "dropped" back onto the board as your own pieces.
    This means pieces are never truly removed from the game.
  - Most pieces can "promote" when reaching the opponent's territory
    (the furthest 3 ranks), gaining new movement abilities.
  - The game has two players: Black (Sente, moves first) and White (Gote).

=== Piece Types ===

Shogi has 8 base piece types. All except King and Gold can promote:
  Pawn (歩)    → promotes to Tokin (と)     — moves like Gold
  Lance (香)   → promotes to ProLance (成香)  — moves like Gold
  Knight (桂)  → promotes to ProKnight (成桂) — moves like Gold
  Silver (銀)  → promotes to ProSilver (成銀) — moves like Gold
  Gold (金)    → cannot promote
  Bishop (角)  → promotes to Horse (馬)      — Bishop + King movement
  Rook (飛)    → promotes to Dragon (龍)     — Rook + King movement
  King (玉/王) → cannot promote

For NNUE, we treat King separately (its position defines the "perspective"),
so we have 13 non-king piece types (7 base + 6 promoted).
"""

# ============================================================================
# Board Dimensions
# ============================================================================

NUM_RANKS = 9   # Rows: numbered 1-9, top (White's side) to bottom (Black's side)
NUM_FILES = 9   # Columns: numbered 9-1, right to left from Black's perspective
NUM_SQUARES = NUM_RANKS * NUM_FILES  # 81 total squares

# Square indexing: squares are numbered 0-80, row by row from top-left.
#
#   File:  9  8  7  6  5  4  3  2  1     (traditional Shogi notation)
#   Index: 0  1  2  3  4  5  6  7  8     Rank 1 (White's back rank)
#          9  10 11 12 13 14 15 16 17    Rank 2
#          ...
#          72 73 74 75 76 77 78 79 80    Rank 9 (Black's back rank)

# ============================================================================
# Piece Type IDs (excluding King)
# ============================================================================
# These integer IDs are used throughout the codebase to identify piece types.
# King is excluded because in NNUE, the king position is used as a separate
# input dimension (see "king buckets" below), not as a regular piece feature.

PAWN = 0
LANCE = 1
KNIGHT = 2
SILVER = 3
GOLD = 4
BISHOP = 5
ROOK = 6
PRO_PAWN = 7      # Tokin (と) — promoted Pawn
PRO_LANCE = 8     # Nari-Kyo (成香) — promoted Lance
PRO_KNIGHT = 9    # Nari-Kei (成桂) — promoted Knight
PRO_SILVER = 10   # Nari-Gin (成銀) — promoted Silver
HORSE = 11        # Uma (馬) — promoted Bishop (Bishop + King movement)
DRAGON = 12       # Ryu (龍) — promoted Rook (Rook + King movement)
NUM_PIECE_TYPES = 13  # 7 base + 6 promoted (excluding King)

KING = 13  # Special ID, only used during SFEN parsing to identify kings

# ============================================================================
# Colors (Players)
# ============================================================================

NUM_COLORS = 2
BLACK = 0  # Sente (先手) — the player who moves first (like White in chess)
WHITE = 1  # Gote (後手) — the player who moves second (like Black in chess)

# ============================================================================
# SFEN Character Mapping
# ============================================================================
# SFEN (Shogi Forsyth-Edwards Notation) uses single characters for pieces.
# Uppercase = Black's pieces, lowercase = White's pieces.
# Promoted pieces are prefixed with '+' (e.g., "+P" = promoted Pawn).
#
# Example SFEN for the starting position:
#   "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
#    ^-- White's pieces (lowercase)         ^-- Black's pieces (uppercase)
#
# The four fields are: board / side_to_move / hand_pieces / move_number

SFEN_PIECE_MAP = {
    'P': PAWN, 'L': LANCE, 'N': KNIGHT, 'S': SILVER,
    'G': GOLD, 'B': BISHOP, 'R': ROOK, 'K': KING,
}
SFEN_PROMOTED_MAP = {
    'P': PRO_PAWN, 'L': PRO_LANCE, 'N': PRO_KNIGHT,
    'S': PRO_SILVER, 'B': HORSE, 'R': DRAGON,
}

# ============================================================================
# Hand Pieces (持ち駒)
# ============================================================================
# In Shogi, captured pieces go to your "hand" and can be dropped back onto
# the board on a future turn. Only unpromoted pieces can be in hand (promoted
# pieces revert to their base form when captured).
#
# Gold cannot be promoted, so it's just "Gold" in hand.
# King can never be captured, so it's never in hand.

HAND_PIECE_TYPES = [PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK]

# Maximum possible count of each piece type in hand.
# These are determined by the total number of each piece in a Shogi set:
#   Pawns: 18 (9 per side), Lance: 4 (2 per side), etc.
HAND_PIECE_MAX = {
    PAWN: 18, LANCE: 4, KNIGHT: 4, SILVER: 4,
    GOLD: 4, BISHOP: 2, ROOK: 2,
}

# Total number of hand-piece features per side.
# We use a "thermometer" encoding: if you have 3 Pawns in hand,
# features at indices 0, 1, 2 are active (out of 18 possible Pawn slots).
# Sum of all max counts: 18+4+4+4+4+2+2 = 38
HAND_FEATURES_PER_SIDE = sum(HAND_PIECE_MAX.values())  # 38

# ============================================================================
# HalfKAv2 Feature Dimensions
# ============================================================================
#
# === What is HalfKAv2? ===
#
# "HalfKA" stands for "Half – King – All-pieces":
#   - "Half": The board is evaluated from two perspectives (one per player).
#             Each half independently produces a feature vector.
#   - "K":    The King's position is used as a key to select which set of
#             weights to use (called "king buckets").
#   - "A":    All other pieces on the board are encoded as features.
#
# "v2" refers to the second version of this architecture, which includes
# improvements to how king adjacency patterns are handled.
#
# The core insight: a Pawn on square 5 means something very different
# depending on whether YOUR King is on square 1 vs square 80. King buckets
# let the network learn different piece-value patterns for different king
# positions.
#
# === Feature Index Layout ===
#
# For each perspective (Black's view, White's view), the features are:
#
#   Board features (pieces on the board, excluding kings):
#     index = piece_type * NUM_COLORS * NUM_SQUARES + color * NUM_SQUARES + square
#     Total: 13 piece_types × 2 colors × 81 squares = 2,106
#
#   Hand features (pieces in hand):
#     Encoded per-side using thermometer coding.
#     Total: 38 features × 2 sides = 76
#
#   Combined per perspective: 2,106 + 76 = 2,182

BOARD_FEATURE_DIM = NUM_PIECE_TYPES * NUM_COLORS * NUM_SQUARES  # 2,106
HALF_FEATURE_DIM = BOARD_FEATURE_DIM + HAND_FEATURES_PER_SIDE * NUM_COLORS  # 2,182

# ============================================================================
# King Buckets
# ============================================================================
# Instead of having separate weights for all 81 possible king squares
# (which would make the weight table enormous), we group king squares
# into "buckets". Squares in the same bucket share weights.
#
# We use 9 buckets, one per rank. This captures the most important
# axis of variation: how far advanced the king is.
#
# Why ranks? In Shogi, the king's rank (how far forward/back it is)
# dramatically affects piece values. A Silver near a castled king on
# the back rank is very different from a Silver near an advancing king.

NUM_KING_BUCKETS = 9


def king_square_to_bucket(square: int) -> int:
    """Map a king square (0-80) to a bucket index (0-8).

    We bucket by rank (row), so all squares on the same rank
    map to the same bucket.

    Args:
        square: Square index (0 = top-left, 80 = bottom-right).

    Returns:
        Bucket index from 0 (rank 1, White's back rank) to 8 (rank 9, Black's back rank).
    """
    rank = square // NUM_FILES
    return rank


def mirror_file(square: int) -> int:
    """Mirror a square horizontally for left-right symmetry.

    Shogi is symmetrical along the vertical center line (file 5).
    By mirroring, we treat "king on file 1" the same as "king on file 9",
    reducing the number of distinct feature patterns the network must learn.

    Specifically: if the square is on the right half of the board
    (files 5-8 in 0-indexed, i.e., files 1-4 in Shogi notation),
    we mirror it to the left half.

    Example:
        Square at (rank=0, file=7) → mirrored to (rank=0, file=1)
        Square at (rank=0, file=2) → unchanged (already on left half)

    Args:
        square: Square index (0-80).

    Returns:
        The (possibly mirrored) square index.
    """
    rank = square // NUM_FILES
    file = square % NUM_FILES
    # Files 0-4 are the left half (including center), 5-8 are the right half.
    # Mirror the right half to the left.
    if file > NUM_FILES // 2:
        file = NUM_FILES - 1 - file
    return rank * NUM_FILES + file


# Total feature transformer input dimension.
# Each king bucket has its own copy of the feature set, so:
#   9 buckets × 2,182 features = 19,638 total input dimensions
# (In practice, only one bucket is active per position, so the input is very sparse.)
FEATURE_TRANSFORMER_INPUT_DIM = NUM_KING_BUCKETS * HALF_FEATURE_DIM  # 19,638
