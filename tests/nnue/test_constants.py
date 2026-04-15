"""Tests for shogi_dojo.nnue.constants — board geometry helpers."""

from shogi_dojo.nnue.constants import (
    NUM_FILES,
    NUM_SQUARES,
    king_square_to_bucket,
    mirror_file,
)


# ---------------------------------------------------------------------------
# king_square_to_bucket
# ---------------------------------------------------------------------------

class TestKingSquareToBucket:
    """Each rank (row) maps to one bucket (0-8)."""

    def test_first_rank_returns_bucket_0(self):
        # Squares 0-8 are rank 0 (White's back rank).
        for sq in range(0, 9):
            assert king_square_to_bucket(sq) == 0

    def test_last_rank_returns_bucket_8(self):
        # Squares 72-80 are rank 8 (Black's back rank).
        for sq in range(72, 81):
            assert king_square_to_bucket(sq) == 8

    def test_center_square(self):
        # Square 40 is rank 4 (middle of the board).
        assert king_square_to_bucket(40) == 4

    def test_all_squares_in_valid_range(self):
        for sq in range(NUM_SQUARES):
            bucket = king_square_to_bucket(sq)
            assert 0 <= bucket <= 8


# ---------------------------------------------------------------------------
# mirror_file
# ---------------------------------------------------------------------------

class TestMirrorFile:
    """Left-right mirroring: right-half squares are flipped to the left half."""

    def test_left_half_unchanged(self):
        # Files 0-4 (left half including center) should not change.
        for rank in range(9):
            for file in range(5):  # files 0,1,2,3,4
                sq = rank * NUM_FILES + file
                assert mirror_file(sq) == sq

    def test_right_half_mirrored(self):
        # File 8 mirrors to file 0, file 7 to file 1, file 6 to file 2, file 5 to file 3.
        for rank in range(9):
            for file in range(5, 9):
                sq = rank * NUM_FILES + file
                expected_file = NUM_FILES - 1 - file
                expected = rank * NUM_FILES + expected_file
                assert mirror_file(sq) == expected

    def test_corners(self):
        assert mirror_file(0) == 0      # top-left → unchanged
        assert mirror_file(8) == 0      # top-right → top-left
        assert mirror_file(72) == 72    # bottom-left → unchanged
        assert mirror_file(80) == 72    # bottom-right → bottom-left
