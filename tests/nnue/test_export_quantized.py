"""Tests for scripts/nnue/export_nnue_quantized.py — quantized NNUE export."""

import struct
import tempfile

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "nnue"))
from export_nnue_quantized import (
    FT_SCALE,
    MAGIC,
    OUTPUT_SCALE,
    VERSION,
    WEIGHT_SCALE,
    export_quantized,
    quantize_bias_i32,
    quantize_ft,
    quantize_weight_i8,
)

from shogi_dojo.nnue.model import NnueNetwork


# ---------------------------------------------------------------------------
# Quantization functions
# ---------------------------------------------------------------------------

class TestQuantizeFt:
    def test_output_dtype_is_int16(self):
        t = torch.tensor([0.5, -0.3, 1.0])
        result = quantize_ft(t, FT_SCALE)
        assert result.dtype == np.int16

    def test_scaling(self):
        t = torch.tensor([1.0, 0.0, -1.0])
        result = quantize_ft(t, FT_SCALE)
        assert result[0] == 127
        assert result[1] == 0
        assert result[2] == -127

    def test_clipping(self):
        t = torch.tensor([300.0, -300.0])
        result = quantize_ft(t, FT_SCALE)
        assert result[0] == 32767   # int16 max
        assert result[1] == -32768  # int16 min


class TestQuantizeWeightI8:
    def test_output_dtype_is_int8(self):
        t = torch.tensor([0.5, -0.3])
        result = quantize_weight_i8(t, WEIGHT_SCALE)
        assert result.dtype == np.int8

    def test_scaling(self):
        t = torch.tensor([1.0, 0.0, -1.0])
        result = quantize_weight_i8(t, WEIGHT_SCALE)
        assert result[0] == 64
        assert result[1] == 0
        assert result[2] == -64

    def test_clipping(self):
        t = torch.tensor([10.0, -10.0])
        result = quantize_weight_i8(t, WEIGHT_SCALE)
        assert result[0] == 127   # int8 max
        assert result[1] == -128  # int8 min


class TestQuantizeBiasI32:
    def test_output_dtype_is_int32(self):
        t = torch.tensor([0.5])
        result = quantize_bias_i32(t, FT_SCALE, WEIGHT_SCALE)
        assert result.dtype == np.int32

    def test_combined_scaling(self):
        t = torch.tensor([1.0])
        result = quantize_bias_i32(t, FT_SCALE, WEIGHT_SCALE)
        expected = FT_SCALE * WEIGHT_SCALE  # 127 * 64 = 8128
        assert result[0] == expected


# ---------------------------------------------------------------------------
# Full export round-trip
# ---------------------------------------------------------------------------

class TestExportQuantized:
    def test_export_produces_file(self):
        # Create a small model and save a checkpoint.
        model = NnueNetwork(ft_input_dim=100, ft_out_dim=8, l1_dim=4, l2_dim=4)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as ckpt_file:
            torch.save({"model_state_dict": model.state_dict()}, ckpt_file.name)
            ckpt_path = ckpt_file.name

        with tempfile.NamedTemporaryFile(suffix=".nnue", delete=False) as out_file:
            out_path = out_file.name

        try:
            export_quantized(ckpt_path, out_path)

            # Verify the file was created and has content.
            import os
            assert os.path.isfile(out_path)
            assert os.path.getsize(out_path) > 0

            # Verify header.
            with open(out_path, "rb") as f:
                magic, version = struct.unpack("<II", f.read(8))
                assert magic == MAGIC
                assert version == VERSION

                ft_in, ft_out, l1, l2 = struct.unpack("<IIII", f.read(16))
                assert ft_in == 100
                assert ft_out == 8
                assert l1 == 4
                assert l2 == 4

                ft_s, w_s, o_s = struct.unpack("<III", f.read(12))
                assert ft_s == FT_SCALE
                assert w_s == WEIGHT_SCALE
                assert o_s == OUTPUT_SCALE
        finally:
            os.unlink(ckpt_path)
            os.unlink(out_path)

    def test_quantized_smaller_than_float32(self):
        """Quantized export should be smaller than a float32 export of the same model."""
        model = NnueNetwork(ft_input_dim=100, ft_out_dim=8, l1_dim=4, l2_dim=4)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as ckpt_file:
            torch.save({"model_state_dict": model.state_dict()}, ckpt_file.name)
            ckpt_path = ckpt_file.name

        with tempfile.NamedTemporaryFile(suffix=".nnue", delete=False) as out_file:
            out_path = out_file.name

        try:
            export_quantized(ckpt_path, out_path)

            import os
            quantized_size = os.path.getsize(out_path)
            # float32 size for same model (rough estimate).
            float32_param_bytes = sum(
                p.numel() * 4 for p in model.parameters()
            )
            assert quantized_size < float32_param_bytes
        finally:
            os.unlink(ckpt_path)
            os.unlink(out_path)
