#!/usr/bin/env python3
"""Export a trained NNUE checkpoint to quantized binary format.

Quantizes float32 weights to integers for fast inference in a Shogi engine:
  - Feature Transformer: int16 weights/bias, scaled by 127
  - Hidden layers (L1, L2): int8 weights, int32 bias, weight scale = 64
  - Output layer (L3): int8 weights, int32 bias, output scale = 600

The scaling factors follow the Stockfish NNUE convention:
  https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html

Usage:
    python scripts/export_nnue_quantized.py checkpoints/best_model.pt model_q.nnue
"""

from __future__ import annotations

import struct
import sys

import numpy as np
import torch

from shogi_dojo.model import NnueNetwork

# Binary format identifiers.
MAGIC = 0x4E4E5545  # "NNUE" in ASCII
VERSION = 2          # Version 2 = quantized format (version 1 = float32)

# Quantization scale factors.
# These map the ClippedReLU output range [0, 1] to integer ranges.
FT_SCALE = 127       # Feature Transformer: float [0,1] -> int16 [0,127]
WEIGHT_SCALE = 64    # Hidden layer weights: float [-1,1] -> int8 [-64,64]
OUTPUT_SCALE = 600   # Output layer: maps network output to centipawns


def quantize_ft(tensor: torch.Tensor, scale: int) -> np.ndarray:
    """Quantize Feature Transformer weights/bias to int16."""
    arr = tensor.detach().cpu().float().numpy()
    quantized = np.clip(np.round(arr * scale), -32768, 32767).astype(np.int16)
    return quantized


def quantize_weight_i8(tensor: torch.Tensor, scale: int) -> np.ndarray:
    """Quantize hidden layer weights to int8."""
    arr = tensor.detach().cpu().float().numpy()
    quantized = np.clip(np.round(arr * scale), -128, 127).astype(np.int8)
    return quantized


def quantize_bias_i32(tensor: torch.Tensor, input_scale: int, weight_scale: int) -> np.ndarray:
    """Quantize hidden layer bias to int32.

    Bias must be scaled by (input_scale * weight_scale) to match the
    accumulated scale of the matrix multiplication output.
    """
    arr = tensor.detach().cpu().float().numpy()
    combined_scale = input_scale * weight_scale
    quantized = np.clip(
        np.round(arr * combined_scale),
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
    ).astype(np.int32)
    return quantized


def export_quantized(checkpoint_path: str, output_path: str):
    """Load a checkpoint and write quantized weights to binary format."""
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    # Infer model dimensions from the saved weight shapes.
    ft_input_dim = state_dict["ft.weight"].shape[0]
    ft_out_dim = state_dict["ft.weight"].shape[1]
    l1_dim = state_dict["l1.weight"].shape[0]
    l2_dim = state_dict["l2.weight"].shape[0]

    model = NnueNetwork(
        ft_input_dim=ft_input_dim,
        ft_out_dim=ft_out_dim,
        l1_dim=l1_dim,
        l2_dim=l2_dim,
    )
    model.load_state_dict(state_dict)

    print("Model dimensions:")
    print(f"  FT: {ft_input_dim} x {ft_out_dim}")
    print(f"  L1: {ft_out_dim * 2} -> {l1_dim}")
    print(f"  L2: {l1_dim} -> {l2_dim}")
    print(f"  L3: {l2_dim} -> 1")
    print()
    print("Quantization scales:")
    print(f"  FT:     {FT_SCALE} (int16)")
    print(f"  Weight: {WEIGHT_SCALE} (int8)")
    print(f"  Output: {OUTPUT_SCALE} (centipawns)")

    with open(output_path, "wb") as f:
        # Header: magic, version, dimensions, scale factors.
        f.write(struct.pack("<II", MAGIC, VERSION))
        f.write(struct.pack("<IIII", ft_input_dim, ft_out_dim, l1_dim, l2_dim))
        f.write(struct.pack("<III", FT_SCALE, WEIGHT_SCALE, OUTPUT_SCALE))

        # Feature Transformer (int16).
        f.write(quantize_ft(model.ft.weight, FT_SCALE).tobytes())
        f.write(quantize_ft(model.ft.bias, FT_SCALE).tobytes())

        # L1: int8 weights, int32 bias.
        # Input to L1 is FT output scaled by FT_SCALE.
        f.write(quantize_weight_i8(model.l1.weight, WEIGHT_SCALE).tobytes())
        f.write(quantize_bias_i32(model.l1.bias, FT_SCALE, WEIGHT_SCALE).tobytes())

        # L2: int8 weights, int32 bias.
        # Input to L2 is L1 output, which after ClippedReLU is re-scaled to FT_SCALE.
        f.write(quantize_weight_i8(model.l2.weight, WEIGHT_SCALE).tobytes())
        f.write(quantize_bias_i32(model.l2.bias, FT_SCALE, WEIGHT_SCALE).tobytes())

        # L3 (output): int8 weights, int32 bias.
        f.write(quantize_weight_i8(model.l3.weight, WEIGHT_SCALE).tobytes())
        f.write(quantize_bias_i32(model.l3.bias, FT_SCALE, WEIGHT_SCALE).tobytes())

    file_size = open(output_path, "rb").seek(0, 2)
    print(f"\nExported to {output_path} ({file_size:,} bytes)")

    # Report compression ratio vs float32 export.
    float32_size = (
        (ft_input_dim * ft_out_dim + ft_out_dim) * 4  # FT: float32
        + (l1_dim * ft_out_dim * 2 + l1_dim) * 4      # L1: float32
        + (l2_dim * l1_dim + l2_dim) * 4               # L2: float32
        + (l2_dim + 1) * 4                             # L3: float32
        + 6 * 4                                         # Header
    )
    print(f"  Float32 equivalent: ~{float32_size:,} bytes")
    print(f"  Compression ratio:  {float32_size / file_size:.1f}x")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <checkpoint.pt> <output.nnue>")
        sys.exit(1)
    export_quantized(sys.argv[1], sys.argv[2])
