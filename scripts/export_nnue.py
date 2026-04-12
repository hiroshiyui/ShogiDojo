#!/usr/bin/env python3
"""Export a trained NNUE checkpoint to a binary format for the Rust gensfen tool.

Usage:
    python scripts/export_nnue.py checkpoints/best_model.pt model.nnue

The binary format is a simple sequence of little-endian values:
    - Header: magic (u32), version (u32), dimensions (4 × u32)
    - Weights and biases: contiguous float32 arrays

This avoids requiring PyTorch or Python in the Rust tool.
"""

import struct
import sys

import torch

from shogi_dojo.model import NnueNetwork

# Binary format magic number: "NNUE" in ASCII
MAGIC = 0x4E4E5545
VERSION = 1


def export(checkpoint_path: str, output_path: str):
    """Load a checkpoint and write weights to binary format."""
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    # Reconstruct model to get dimensions
    model = NnueNetwork()
    model.load_state_dict(state_dict)

    ft_input_dim = model.ft.weight.shape[0]
    ft_out_dim = model.ft.weight.shape[1]
    l1_dim = model.l1.weight.shape[0]
    l2_dim = model.l2.weight.shape[0]

    print(f"Model dimensions:")
    print(f"  FT: {ft_input_dim} × {ft_out_dim}")
    print(f"  L1: {ft_out_dim * 2} → {l1_dim}")
    print(f"  L2: {l1_dim} → {l2_dim}")
    print(f"  L3: {l2_dim} → 1")

    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<II", MAGIC, VERSION))
        f.write(struct.pack("<IIII", ft_input_dim, ft_out_dim, l1_dim, l2_dim))

        # Feature Transformer weights and bias
        # weight shape: (ft_input_dim, ft_out_dim) — already row-major
        _write_tensor(f, model.ft.weight)
        _write_tensor(f, model.ft.bias)

        # Linear layers: PyTorch stores weight as (out_features, in_features)
        _write_tensor(f, model.l1.weight)
        _write_tensor(f, model.l1.bias)
        _write_tensor(f, model.l2.weight)
        _write_tensor(f, model.l2.bias)
        _write_tensor(f, model.l3.weight)
        _write_tensor(f, model.l3.bias)

    file_size = open(output_path, "rb").seek(0, 2)
    print(f"Exported to {output_path} ({file_size:,} bytes)")


def _write_tensor(f, tensor: torch.Tensor):
    """Write a tensor as contiguous little-endian float32."""
    data = tensor.detach().cpu().contiguous().float().numpy()
    f.write(data.tobytes())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <checkpoint.pt> <output.nnue>")
        sys.exit(1)
    export(sys.argv[1], sys.argv[2])
