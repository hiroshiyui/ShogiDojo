#!/usr/bin/env python3
"""Export an AlphaZero/Bonanza checkpoint to TorchScript.

TorchScript is PyTorch's own portable model format. The exported ``.ts``
file can be loaded by libtorch (the C++ PyTorch runtime) without
requiring Python or the original model class definition. Use this when
embedding the network in a C++ application that already depends on
libtorch.

For mobile or web deployment, prefer ONNX (see ``export_onnx.py``);
TorchScript would force the app to ship ~100 MB of libtorch, while
ONNX Runtime Mobile is ~5 MB.

Both AlphaZero-trained and Bonanza-trained checkpoints are supported —
the network architecture (``AlphaZeroNetwork``) is identical and state
dicts are byte-for-byte interchangeable.

Usage:
    python scripts/alphazero/export_torchscript.py \\
        --checkpoint checkpoints/bonanza/bonanza_epoch_0050.pt \\
        --output bonanza.ts

    # If the checkpoint was trained with non-default architecture:
    python scripts/alphazero/export_torchscript.py \\
        --checkpoint model.pt --output model.ts \\
        --num-blocks 8 --num-filters 128
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings

import torch

from shogi_dojo.alphazero.encoding import NUM_INPUT_CHANNELS
from shogi_dojo.alphazero.network import AlphaZeroNetwork
from shogi_dojo.nnue.constants import NUM_FILES, NUM_RANKS

logger = logging.getLogger(__name__)


def export_torchscript(
    checkpoint_path: str,
    output_path: str,
    *,
    num_blocks: int = 4,
    num_filters: int = 64,
    value_head_hidden: int = 64,
) -> None:
    """Trace the network and save as TorchScript.

    The traced module accepts a tensor of shape
    ``(B, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES) == (B, 45, 9, 9)`` and
    returns a tuple ``(policy_logits, value)``. The batch dimension is
    dynamic — any batch size works at inference time, even though we
    trace with batch_size=1.

    Verifies that the traced module's outputs match the original
    PyTorch module's outputs within a small tolerance before writing
    the file.
    """
    net = AlphaZeroNetwork(
        num_blocks=num_blocks,
        num_filters=num_filters,
        value_head_hidden=value_head_hidden,
    )
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # Trace with a single-batch dummy input. eval() mode ensures
    # BatchNorm uses its tracked running statistics rather than the
    # batch's own stats — required for correct single-batch inference.
    dummy = torch.randn(1, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
    # Recent PyTorch versions emit DeprecationWarnings urging callers to
    # migrate to ``torch.compile`` / ``torch.export``. Those alternatives
    # are different paradigms (Python-level graph capture, not a
    # libtorch-loadable archive) and do not replace TorchScript for C++
    # deployment. Silence the noise locally rather than rejecting the
    # API surface that this script intentionally targets.
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.jit")
        traced = torch.jit.trace(net, dummy)

        # Round-trip check: the traced module must reproduce the
        # original outputs. atol=1e-5 catches genuine differences while
        # tolerating floating-point reordering by the JIT compiler.
        ref_policy, ref_value = net(dummy)
        out_policy, out_value = traced(dummy)
        if not torch.allclose(ref_policy, out_policy, atol=1e-5):
            raise RuntimeError("TorchScript policy output diverged from PyTorch")
        if not torch.allclose(ref_value, out_value, atol=1e-5):
            raise RuntimeError("TorchScript value output diverged from PyTorch")

        traced.save(output_path)

    size_kb = os.path.getsize(output_path) / 1024
    logger.info("Exported TorchScript: %s (%.1f KB)", output_path, size_kb)
    logger.info(
        "  Input:  (B, %d, %d, %d)", NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES,
    )
    logger.info("  Output: (policy_logits (B, 139, 9, 9), value (B, 1))")


def main():
    parser = argparse.ArgumentParser(
        description="Export an AlphaZero/Bonanza checkpoint to TorchScript.",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a .pt checkpoint produced by AlphaZeroTrainer or BonanzaTrainer.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the TorchScript file (.ts).",
    )
    parser.add_argument(
        "--num-blocks", type=int, default=4,
        help="Number of residual blocks (must match the checkpoint).",
    )
    parser.add_argument(
        "--num-filters", type=int, default=64,
        help="Number of conv filters per block (must match the checkpoint).",
    )
    parser.add_argument(
        "--value-head-hidden", type=int, default=64,
        help="Value head hidden size (must match the checkpoint).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    export_torchscript(
        args.checkpoint, args.output,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        value_head_hidden=args.value_head_hidden,
    )


if __name__ == "__main__":
    main()
