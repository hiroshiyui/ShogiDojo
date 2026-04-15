"""Tests for scripts/validate_sfen.py — SFEN training data validation."""

import os
import tempfile

# Import from the script (add scripts/ to path).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from validate_sfen import validate_file, validate_line


# A valid tab-separated line (gensfen output format).
VALID_TAB = "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\t100\t1.0"

# A valid space-separated line (alternative format).
VALID_SPACE = "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1 100 1.0"


# ---------------------------------------------------------------------------
# validate_line — valid lines
# ---------------------------------------------------------------------------

class TestValidateLineValid:
    def test_tab_separated(self):
        sfen, score, result, error = validate_line(VALID_TAB)
        assert error is None
        assert score == 100.0
        assert result == 1.0
        assert "b" in sfen

    def test_space_separated(self):
        sfen, score, result, error = validate_line(VALID_SPACE)
        assert error is None
        assert score == 100.0
        assert result == 1.0

    def test_without_sfen_prefix(self):
        line = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\t50\t0.5"
        sfen, score, result, error = validate_line(line)
        assert error is None
        assert score == 50.0
        assert result == 0.5

    def test_negative_score(self):
        line = "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 2\t-300\t0.0"
        sfen, score, result, error = validate_line(line)
        assert error is None
        assert score == -300.0
        assert result == 0.0

    def test_draw_result(self):
        line = "sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1\t0\t0.5"
        _, _, result, error = validate_line(line)
        assert error is None
        assert result == 0.5

    def test_white_to_move(self):
        line = "sfen 4k4/9/9/9/9/9/9/9/4K4 w - 1\t100\t1.0"
        sfen, _, _, error = validate_line(line)
        assert error is None
        assert "w" in sfen


# ---------------------------------------------------------------------------
# validate_line — malformed lines
# ---------------------------------------------------------------------------

class TestValidateLineMalformed:
    def test_empty_line(self):
        _, _, _, error = validate_line("")
        assert error == "empty line"

    def test_whitespace_only(self):
        _, _, _, error = validate_line("   \n")
        assert error == "empty line"

    def test_too_few_fields_space(self):
        _, _, _, error = validate_line("sfen board side hand")
        assert error is not None
        assert "fields" in error

    def test_too_few_fields_tab(self):
        _, _, _, error = validate_line("sfen board\t100")
        assert error is not None

    def test_invalid_side_to_move(self):
        line = "sfen 4k4/9/9/9/9/9/9/9/4K4 x - 1\t100\t1.0"
        _, _, _, error = validate_line(line)
        assert error is not None
        assert "side-to-move" in error

    def test_non_numeric_score(self):
        line = "sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1\tabc\t1.0"
        _, _, _, error = validate_line(line)
        assert error is not None
        assert "score" in error

    def test_inf_score(self):
        line = "sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1\tinf\t1.0"
        _, _, _, error = validate_line(line)
        assert error is not None
        assert "non-finite" in error

    def test_nan_score(self):
        line = "sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1\tnan\t1.0"
        _, _, _, error = validate_line(line)
        assert error is not None
        assert "non-finite" in error

    def test_invalid_result(self):
        line = "sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1\t100\t0.7"
        _, _, _, error = validate_line(line)
        assert error is not None
        assert "result" in error

    def test_non_numeric_result(self):
        line = "sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1\t100\twin"
        _, _, _, error = validate_line(line)
        assert error is not None
        assert "result" in error

    def test_garbage(self):
        _, _, _, error = validate_line("completely garbage data here")
        assert error is not None


# ---------------------------------------------------------------------------
# validate_file — integration tests
# ---------------------------------------------------------------------------

def _write_temp(lines: list[str]) -> str:
    """Write lines to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".sfen", delete=False)
    for line in lines:
        f.write(line + "\n")
    f.close()
    return f.name


class TestValidateFile:
    def test_all_valid(self):
        path = _write_temp([VALID_TAB, VALID_TAB.replace("100", "-50").replace("1.0", "0.0")])
        try:
            stats = validate_file(Path(path), check_duplicates=False, verbose=True)
            assert stats["valid"] == 2
            assert stats["malformed"] == 0
            assert len(stats["scores"]) == 2
        finally:
            os.unlink(path)

    def test_mixed_valid_and_invalid(self):
        path = _write_temp([
            VALID_TAB,
            "garbage line",
            "",
            VALID_TAB.replace("100", "200").replace("1.0", "0.5"),
        ])
        try:
            stats = validate_file(Path(path), check_duplicates=False, verbose=True)
            assert stats["valid"] == 2
            assert stats["malformed"] == 1
            assert stats["empty"] == 1
            assert stats["total_lines"] == 4
        finally:
            os.unlink(path)

    def test_empty_file(self):
        path = _write_temp([])
        try:
            stats = validate_file(Path(path), check_duplicates=False, verbose=True)
            assert stats["valid"] == 0
            assert stats["total_lines"] == 0
        finally:
            os.unlink(path)

    def test_nonexistent_file(self):
        stats = validate_file(Path("/nonexistent/file.sfen"), check_duplicates=False, verbose=True)
        assert stats["malformed"] == -1

    def test_duplicate_detection(self):
        path = _write_temp([VALID_TAB, VALID_TAB, VALID_TAB])
        try:
            stats = validate_file(Path(path), check_duplicates=True, verbose=True)
            assert stats["valid"] == 3
            assert stats["duplicates"] == 2  # 2nd and 3rd are duplicates
        finally:
            os.unlink(path)

    def test_no_duplicates_when_disabled(self):
        path = _write_temp([VALID_TAB, VALID_TAB])
        try:
            stats = validate_file(Path(path), check_duplicates=False, verbose=True)
            assert stats["duplicates"] == 0  # Not checked
        finally:
            os.unlink(path)

    def test_result_counts(self):
        lines = [
            "sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1\t100\t1.0",
            "sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1\t-100\t0.0",
            "sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1\t0\t0.5",
            "sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1\t50\t1.0",
        ]
        path = _write_temp(lines)
        try:
            stats = validate_file(Path(path), check_duplicates=False, verbose=True)
            assert stats["results"][1.0] == 2
            assert stats["results"][0.0] == 1
            assert stats["results"][0.5] == 1
        finally:
            os.unlink(path)

    def test_score_values_collected(self):
        lines = [
            "sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1\t100\t1.0",
            "sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1\t-200\t0.0",
            "sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1\t0\t0.5",
        ]
        path = _write_temp(lines)
        try:
            stats = validate_file(Path(path), check_duplicates=False, verbose=True)
            assert stats["scores"] == [100.0, -200.0, 0.0]
        finally:
            os.unlink(path)

    def test_errors_capped_at_20(self):
        lines = ["bad line"] * 30
        path = _write_temp(lines)
        try:
            stats = validate_file(Path(path), check_duplicates=False, verbose=True)
            assert stats["malformed"] == 30
            assert len(stats["errors"]) == 20  # Capped
        finally:
            os.unlink(path)
