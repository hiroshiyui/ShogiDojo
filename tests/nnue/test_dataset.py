"""Tests for shogi_dojo.dataset — training data parsing and batch collation."""

import tempfile
import os

import torch

from shogi_dojo.dataset import SfenDataset, collate_fn


# A valid training data line (starting position, score=100cp, result=1.0 win).
VALID_LINE = (
    "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\t100\t1.0"
)

# A second line with different score/result for collation tests.
VALID_LINE_2 = (
    "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 2\t-50\t0.0"
)


class TestSfenDatasetParseLine:
    """Unit tests for SfenDataset._parse_line (single-line parsing)."""

    def setup_method(self):
        self.ds = SfenDataset.__new__(SfenDataset)
        self.ds.score_scaling = 600.0

    def test_valid_line(self):
        sample = self.ds._parse_line(VALID_LINE)
        assert sample is not None
        assert "stm_indices" in sample
        assert "nstm_indices" in sample
        assert "score" in sample
        assert "result" in sample

    def test_score_scaling(self):
        sample = self.ds._parse_line(VALID_LINE)
        # Raw score is 100, scaling is 600 → 100/600
        expected = torch.tensor(100.0 / 600.0)
        assert torch.isclose(sample["score"], expected)

    def test_result_value(self):
        sample = self.ds._parse_line(VALID_LINE)
        assert sample["result"].item() == 1.0

    def test_indices_are_long_tensors(self):
        sample = self.ds._parse_line(VALID_LINE)
        assert sample["stm_indices"].dtype == torch.long
        assert sample["nstm_indices"].dtype == torch.long

    def test_malformed_line_returns_none(self):
        assert self.ds._parse_line("garbage data") is None
        assert self.ds._parse_line("") is None

    def test_line_without_sfen_prefix(self):
        # Lines without "sfen " prefix should still parse (prefix is optional).
        line = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1 100 1.0"
        sample = self.ds._parse_line(line)
        assert sample is not None


class TestSfenDatasetIter:
    """Integration test: iterate over a small file."""

    def test_iteration(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sfen", delete=False) as f:
            f.write(VALID_LINE + "\n")
            f.write(VALID_LINE_2 + "\n")
            f.write("\n")  # blank line should be skipped
            path = f.name

        try:
            ds = SfenDataset(path)
            samples = list(ds)
            assert len(samples) == 2
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# collate_fn — padding with -1
# ---------------------------------------------------------------------------

class TestCollateFn:
    """Verify that collate_fn pads variable-length indices correctly."""

    def _make_sample(self, stm_len: int, nstm_len: int, score: float, result: float):
        return {
            "stm_indices": torch.arange(stm_len, dtype=torch.long),
            "nstm_indices": torch.arange(nstm_len, dtype=torch.long),
            "score": torch.tensor(score, dtype=torch.float32),
            "result": torch.tensor(result, dtype=torch.float32),
        }

    def test_padding_value_is_minus_one(self):
        batch = [
            self._make_sample(3, 3, 0.1, 1.0),
            self._make_sample(5, 4, 0.2, 0.0),
        ]
        out = collate_fn(batch)
        # First sample's stm_indices has length 3, padded to 5.
        assert out["stm_indices"][0, 3].item() == -1
        assert out["stm_indices"][0, 4].item() == -1
        # Second sample has no padding in stm.
        assert out["stm_indices"][1, 4].item() == 4

    def test_output_shapes(self):
        batch = [
            self._make_sample(3, 4, 0.1, 1.0),
            self._make_sample(5, 2, 0.2, 0.0),
            self._make_sample(4, 3, 0.3, 0.5),
        ]
        out = collate_fn(batch)
        assert out["stm_indices"].shape == (3, 5)   # max stm len = 5
        assert out["nstm_indices"].shape == (3, 4)   # max nstm len = 4
        assert out["score"].shape == (3,)
        assert out["result"].shape == (3,)

    def test_valid_indices_preserved(self):
        batch = [self._make_sample(4, 4, 0.0, 0.0)]
        out = collate_fn(batch)
        assert torch.equal(out["stm_indices"][0], torch.arange(4))

    def test_single_element_batch(self):
        batch = [self._make_sample(3, 2, 0.5, 0.5)]
        out = collate_fn(batch)
        assert out["stm_indices"].shape == (1, 3)
        assert out["nstm_indices"].shape == (1, 2)
        assert out["score"].shape == (1,)

    def test_uniform_lengths_no_padding(self):
        batch = [
            self._make_sample(4, 4, 0.1, 0.0),
            self._make_sample(4, 4, 0.2, 1.0),
        ]
        out = collate_fn(batch)
        # No -1 padding needed when all samples have equal length.
        assert (out["stm_indices"] >= 0).all()


# ---------------------------------------------------------------------------
# SfenDataset._parse_line — additional edge cases
# ---------------------------------------------------------------------------

class TestSfenDatasetParseLineEdgeCases:
    """Edge case coverage for _parse_line."""

    def setup_method(self):
        self.ds = SfenDataset.__new__(SfenDataset)
        self.ds.score_scaling = 600.0

    def test_negative_score(self):
        line = "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\t-300\t0.0"
        sample = self.ds._parse_line(line)
        assert sample is not None
        expected = torch.tensor(-300.0 / 600.0)
        assert torch.isclose(sample["score"], expected)

    def test_draw_result(self):
        line = "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\t0\t0.5"
        sample = self.ds._parse_line(line)
        assert sample is not None
        assert sample["result"].item() == 0.5

    def test_tab_separated_format(self):
        """Training data uses tab separators between sfen, score, and result."""
        line = "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\t100\t1.0"
        sample = self.ds._parse_line(line)
        assert sample is not None

    def test_too_few_fields_returns_none(self):
        # Only 3 SFEN fields, missing score and result.
        assert self.ds._parse_line("sfen board side hand") is None

    def test_invalid_sfen_in_line_returns_none(self):
        # Valid field count but garbage SFEN — sfen_to_features will fail.
        line = "sfen XXXXXX b - 1 100 1.0"
        assert self.ds._parse_line(line) is None


# ---------------------------------------------------------------------------
# SfenDataset — file-level edge cases
# ---------------------------------------------------------------------------

class TestSfenDatasetFile:
    """File I/O and iteration edge cases."""

    def test_file_with_only_blank_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sfen", delete=False) as f:
            f.write("\n\n\n")
            path = f.name
        try:
            ds = SfenDataset(path)
            assert list(ds) == []
        finally:
            os.unlink(path)

    def test_file_with_mixed_valid_and_invalid(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sfen", delete=False) as f:
            f.write(VALID_LINE + "\n")
            f.write("corrupted garbage line\n")
            f.write(VALID_LINE_2 + "\n")
            path = f.name
        try:
            ds = SfenDataset(path)
            samples = list(ds)
            assert len(samples) == 2  # corrupted line silently skipped
        finally:
            os.unlink(path)
