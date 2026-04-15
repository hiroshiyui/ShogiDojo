"""Tests for shogi_dojo.features — SFEN parsing and HalfKAv2 feature extraction."""

import pytest

from shogi_dojo.constants import (
    BLACK,
    DRAGON,
    FEATURE_TRANSFORMER_INPUT_DIM,
    HALF_FEATURE_DIM,
    HORSE,
    NUM_KING_BUCKETS,
    PAWN,
    WHITE,
)
from shogi_dojo.features import _flip_square, parse_sfen, sfen_to_features

# The standard Shogi starting position in SFEN.
STARTPOS = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


# ---------------------------------------------------------------------------
# _flip_square
# ---------------------------------------------------------------------------

class TestFlipSquare:
    def test_corners(self):
        assert _flip_square(0) == 80
        assert _flip_square(80) == 0

    def test_center_unchanged(self):
        assert _flip_square(40) == 40

    def test_double_flip_identity(self):
        for sq in range(81):
            assert _flip_square(_flip_square(sq)) == sq


# ---------------------------------------------------------------------------
# parse_sfen — starting position
# ---------------------------------------------------------------------------

class TestParseSfenStartpos:
    """Verify the starting position is parsed correctly."""

    def setup_method(self):
        self.pieces, self.stm, self.hand, self.bk, self.wk = parse_sfen(STARTPOS)

    def test_side_to_move_is_black(self):
        assert self.stm == BLACK

    def test_king_squares(self):
        # Black's king is on rank 9, file 5 → square index 76 (row 8, col 4).
        assert self.bk == 76
        # White's king is on rank 1, file 5 → square index 4 (row 0, col 4).
        assert self.wk == 4

    def test_piece_count(self):
        # 40 pieces total minus 2 kings = 38 non-king pieces.
        assert len(self.pieces) == 38

    def test_no_hand_pieces(self):
        for color in range(2):
            assert all(c == 0 for c in self.hand[color])

    def test_pawn_count(self):
        pawns = [(pt, c, s) for pt, c, s in self.pieces if pt == PAWN]
        assert len(pawns) == 18  # 9 per side


# ---------------------------------------------------------------------------
# parse_sfen — position with hand pieces
# ---------------------------------------------------------------------------

class TestParseSfenHandPieces:
    """Test parsing a position where pieces are in hand."""

    def test_hand_parsing(self):
        # Black holds 2 Pawns and 1 Bishop; White holds 1 Gold.
        # All hand pieces must be in a single token: uppercase = Black, lowercase = White.
        sfen = "4k4/9/9/9/9/9/9/9/4K4 b 2PBg 1"
        _, stm, hand, _, _ = parse_sfen(sfen)
        assert stm == BLACK
        # Black hand
        assert hand[BLACK][0] == 2   # 2 Pawns (index 0 in HAND_PIECE_TYPES)
        assert hand[BLACK][5] == 1   # 1 Bishop (index 5)
        # White hand
        assert hand[WHITE][4] == 1   # 1 Gold (index 4)

    def test_empty_hand(self):
        sfen = "4k4/9/9/9/9/9/9/9/4K4 b - 1"
        _, _, hand, _, _ = parse_sfen(sfen)
        for color in range(2):
            assert all(c == 0 for c in hand[color])


# ---------------------------------------------------------------------------
# parse_sfen — promoted pieces
# ---------------------------------------------------------------------------

class TestParseSfenPromotion:
    """Test that '+' prefix correctly maps to promoted piece types."""

    def test_promoted_bishop_is_horse(self):
        sfen = "4k4/9/9/9/4+B4/9/9/9/4K4 b - 1"
        pieces, _, _, _, _ = parse_sfen(sfen)
        assert len(pieces) == 1
        assert pieces[0][0] == HORSE  # promoted Bishop = Horse


# ---------------------------------------------------------------------------
# make_feature_indices / sfen_to_features
# ---------------------------------------------------------------------------

class TestFeatureIndices:
    """Verify feature index properties for the starting position."""

    def setup_method(self):
        self.stm_idx, self.nstm_idx, self.stm = sfen_to_features(STARTPOS)

    def test_indices_in_valid_range(self):
        for idx in self.stm_idx + self.nstm_idx:
            assert 0 <= idx < FEATURE_TRANSFORMER_INPUT_DIM

    def test_no_duplicate_indices_asymmetric(self):
        """An asymmetric position (pieces not mirrored left-right) should have
        no duplicate indices. The symmetric starting position *can* have
        duplicates because mirror_file maps right-half pieces onto left-half
        squares that are already occupied."""
        # Bishop on file 2 (left half) — no mirror collision.
        sfen = "4k4/9/9/9/1B7/9/9/9/4K4 b - 1"
        stm, nstm, _ = sfen_to_features(sfen)
        assert len(stm) == len(set(stm))
        assert len(nstm) == len(set(nstm))

    def test_both_perspectives_same_count(self):
        # Same position → same number of active features from each perspective.
        assert len(self.stm_idx) == len(self.nstm_idx)

    def test_nonempty(self):
        # Starting position has 38 board pieces + 0 hand → at least 38 features.
        assert len(self.stm_idx) >= 38

    def test_perspectives_differ_asymmetric(self):
        """STM and NSTM perspectives should differ for an asymmetric position.
        The starting position is fully symmetric (left-right mirror + 180-degree
        rotation maps it to itself), so both perspectives coincide there."""
        # Asymmetric: a single Black Pawn on file 1 (left side).
        sfen = "4k4/9/9/9/P8/9/9/9/4K4 b - 1"
        stm, nstm, _ = sfen_to_features(sfen)
        assert set(stm) != set(nstm)


class TestFeatureIndicesSymmetry:
    """When Black is STM, stm_indices == black_indices.
    When White is STM, stm_indices == white_indices."""

    def test_white_to_move_swaps_perspectives(self):
        sfen_b = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        sfen_w = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1"

        stm_b, nstm_b, _ = sfen_to_features(sfen_b)
        stm_w, nstm_w, _ = sfen_to_features(sfen_w)

        # When side-to-move flips, STM ↔ NSTM should swap.
        assert sorted(stm_b) == sorted(nstm_w)
        assert sorted(nstm_b) == sorted(stm_w)


# ---------------------------------------------------------------------------
# parse_sfen — promoted pieces on board
# ---------------------------------------------------------------------------

class TestParseSfenPromotedPieces:
    """Verify various promoted piece types are parsed correctly."""

    def test_promoted_rook_is_dragon(self):
        sfen = "4k4/9/9/9/4+R4/9/9/9/4K4 b - 1"
        pieces, _, _, _, _ = parse_sfen(sfen)
        assert len(pieces) == 1
        assert pieces[0][0] == DRAGON

    def test_multiple_promoted_pieces(self):
        sfen = "4k4/9/9/9/3+B+R4/9/9/9/4K4 b - 1"
        pieces, _, _, _, _ = parse_sfen(sfen)
        assert len(pieces) == 2
        types = {p[0] for p in pieces}
        assert HORSE in types
        assert DRAGON in types


# ---------------------------------------------------------------------------
# parse_sfen — multi-digit hand counts
# ---------------------------------------------------------------------------

class TestParseSfenLargeHand:
    """Test hand parsing with double-digit piece counts (e.g., 18P)."""

    def test_18_pawns_in_hand(self):
        sfen = "4k4/9/9/9/9/9/9/9/4K4 b 18P 1"
        _, _, hand, _, _ = parse_sfen(sfen)
        assert hand[BLACK][0] == 18  # 18 Pawns

    def test_mixed_hand(self):
        sfen = "4k4/9/9/9/9/9/9/9/4K4 b 2R2B4G4S4N4L18P 1"
        _, _, hand, _, _ = parse_sfen(sfen)
        assert hand[BLACK][0] == 18  # Pawn
        assert hand[BLACK][1] == 4   # Lance
        assert hand[BLACK][2] == 4   # Knight
        assert hand[BLACK][3] == 4   # Silver
        assert hand[BLACK][4] == 4   # Gold
        assert hand[BLACK][5] == 2   # Bishop
        assert hand[BLACK][6] == 2   # Rook


# ---------------------------------------------------------------------------
# parse_sfen — king positions at edges
# ---------------------------------------------------------------------------

class TestParseSfenKingPositions:
    """Test king square extraction for different positions."""

    def test_kings_at_corners(self):
        # Black King at 1a (bottom-left), White King at 9i (top-right).
        sfen = "K8/9/9/9/9/9/9/9/8k b - 1"
        _, _, _, bk, wk = parse_sfen(sfen)
        # K at rank 0, file 0 → square 0
        assert bk == 0 or wk == 0
        # k at rank 8, file 8 → square 80
        assert bk == 80 or wk == 80

    def test_side_to_move_white(self):
        sfen = "4k4/9/9/9/9/9/9/9/4K4 w - 1"
        _, stm, _, _, _ = parse_sfen(sfen)
        assert stm == WHITE


# ---------------------------------------------------------------------------
# Feature indices — hand features via thermometer encoding
# ---------------------------------------------------------------------------

class TestHandFeatures:
    """Verify hand pieces generate thermometer-encoded features."""

    def test_hand_pieces_add_features(self):
        """Position with hand pieces should produce more features than without."""
        sfen_no_hand = "4k4/9/9/9/9/9/9/9/4K4 b - 1"
        sfen_with_hand = "4k4/9/9/9/9/9/9/9/4K4 b 3P 1"
        stm_no, _, _ = sfen_to_features(sfen_no_hand)
        stm_with, _, _ = sfen_to_features(sfen_with_hand)
        # 3 Pawns in hand → 3 extra features (thermometer: indices 0, 1, 2).
        assert len(stm_with) == len(stm_no) + 3

    def test_hand_features_in_bounds(self):
        sfen = "4k4/9/9/9/9/9/9/9/4K4 b 2R2B4G4S4N4L18P 1"
        stm, nstm, _ = sfen_to_features(sfen)
        for idx in stm + nstm:
            assert 0 <= idx < FEATURE_TRANSFORMER_INPUT_DIM


# ---------------------------------------------------------------------------
# Feature indices — king buckets affect indices
# ---------------------------------------------------------------------------

class TestKingBucketEffect:
    """Different king positions should produce different feature indices
    (because the king bucket changes the index offset)."""

    def test_different_king_rank_different_indices(self):
        # Same Pawn, different king rank → different bucket → different indices.
        sfen_a = "4k4/9/9/9/P8/9/9/9/4K4 b - 1"   # King on rank 8
        sfen_b = "4k4/9/9/9/P3K4/9/9/9/9 b - 1"    # King on rank 4
        stm_a, _, _ = sfen_to_features(sfen_a)
        stm_b, _, _ = sfen_to_features(sfen_b)
        # The Pawn's feature index differs because the king bucket differs.
        assert stm_a != stm_b


# ---------------------------------------------------------------------------
# Feature index upper bound (covers every bucket)
# ---------------------------------------------------------------------------

class TestFeatureMaxIndex:
    """The maximum possible feature index should be < FEATURE_TRANSFORMER_INPUT_DIM."""

    def test_max_index_across_positions(self):
        positions = [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "4k4/9/9/9/9/9/9/9/4K4 b 2R2B4G4S4N4L18P 1",
            "4k4/9/4+R4/9/4+B4/9/9/9/4K4 b - 1",
        ]
        max_dim = NUM_KING_BUCKETS * HALF_FEATURE_DIM
        for sfen in positions:
            stm, nstm, _ = sfen_to_features(sfen)
            for idx in stm + nstm:
                assert 0 <= idx < max_dim, f"idx {idx} out of bounds for sfen: {sfen}"


# ---------------------------------------------------------------------------
# parse_sfen — malformed input rejected
# ---------------------------------------------------------------------------

class TestParseSfenValidation:
    """Malformed SFEN strings must raise ValueError.

    The parser is a system boundary (user-supplied text), so it enforces
    basic Shogi invariants to fail loudly instead of silently producing
    nonsensical feature indices.
    """

    def test_too_few_fields(self):
        with pytest.raises(ValueError, match="space-separated fields"):
            parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b")

    def test_invalid_side_to_move(self):
        with pytest.raises(ValueError, match="side-to-move"):
            parse_sfen("4k4/9/9/9/9/9/9/9/4K4 x - 1")

    def test_missing_black_king(self):
        with pytest.raises(ValueError, match="Black King missing"):
            parse_sfen("4k4/9/9/9/9/9/9/9/9 b - 1")

    def test_missing_white_king(self):
        with pytest.raises(ValueError, match="White King missing"):
            parse_sfen("9/9/9/9/9/9/9/9/4K4 b - 1")

    def test_two_black_kings(self):
        with pytest.raises(ValueError, match="more than one Black King"):
            parse_sfen("4k4/9/9/9/9/9/9/4K4/4K4 b - 1")

    def test_board_too_short(self):
        # 8 ranks instead of 9 → total squares != 81.
        with pytest.raises(ValueError, match="expected 81"):
            parse_sfen("4k4/9/9/9/9/9/9/4K4 b - 1")

    def test_board_rank_too_wide(self):
        # A rank with more than 9 squares overruns.
        with pytest.raises(ValueError, match="squares"):
            parse_sfen("4k4/9/9/9/9/9/9/9/4K44 b - 1")

    def test_unknown_piece_char(self):
        with pytest.raises(ValueError, match="unknown piece"):
            parse_sfen("4k4/9/9/9/4Z4/9/9/9/4K4 b - 1")

    def test_promoted_king(self):
        with pytest.raises(ValueError, match="King cannot be promoted"):
            parse_sfen("4k4/9/9/9/4+K4/9/9/9/9 b - 1")

    def test_promoted_gold(self):
        # Gold cannot promote.
        with pytest.raises(ValueError, match="cannot be promoted"):
            parse_sfen("4k4/9/9/9/4+G4/9/9/9/4K4 b - 1")

    def test_hand_contains_promoted_piece(self):
        with pytest.raises(ValueError, match="promoted piece not allowed in hand"):
            parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b +P 1")

    def test_hand_contains_king(self):
        with pytest.raises(ValueError, match="invalid piece"):
            parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b K 1")

    def test_hand_count_exceeds_max(self):
        # Only 18 Pawns exist in a Shogi set.
        with pytest.raises(ValueError, match="exceeds maximum"):
            parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b 19P 1")

    def test_hand_rook_exceeds_max(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b 3R 1")

    def test_hand_piece_listed_twice(self):
        # Re-listing the same piece for the same side is malformed.
        with pytest.raises(ValueError, match="listed twice"):
            parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2P3P 1")

    def test_trailing_digit_in_hand(self):
        with pytest.raises(ValueError, match="trailing digits"):
            parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2P3 1")

    def test_double_promotion_prefix(self):
        with pytest.raises(ValueError, match="doubled '\\+'"):
            parse_sfen("4k4/9/9/9/4++P4/9/9/9/4K4 b - 1")

    def test_promotion_prefix_before_digit(self):
        with pytest.raises(ValueError, match="'\\+' must precede a piece"):
            parse_sfen("4k4/9/9/9/+4P4/9/9/9/4K4 b - 1")
