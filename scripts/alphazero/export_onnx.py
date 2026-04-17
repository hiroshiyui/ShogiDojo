#!/usr/bin/env python3
"""Export an AlphaZero/Bonanza checkpoint to ONNX format.

ONNX (Open Neural Network Exchange) is a portable model format
supported by many runtimes:

- **ONNX Runtime** — Python, C++, Rust (`ort` crate), JavaScript (web)
- **Apple Core ML** — convert via `coremltools` for iOS deployment
- **TensorFlow Lite** — convert via `tf2onnx` (reverse) for Android
- **NVIDIA TensorRT** — for high-throughput GPU inference

This is the recommended export path for mobile or web deployment of
the Bonanza-trained network (e.g. a Shogi sensei app), since ONNX
Runtime Mobile is ~5 MB versus ~100 MB for libtorch.

Both AlphaZero-trained and Bonanza-trained checkpoints are supported —
the network architecture (``AlphaZeroNetwork``) is identical and state
dicts are byte-for-byte interchangeable.

Usage:
    python scripts/alphazero/export_onnx.py \\
        --checkpoint checkpoints/bonanza/bonanza_epoch_0050.pt \\
        --output bonanza.onnx

    # If the checkpoint was trained with non-default architecture:
    python scripts/alphazero/export_onnx.py \\
        --checkpoint model.pt --output model.onnx \\
        --num-blocks 8 --num-filters 128

If ``onnxruntime`` is installed, the export is round-trip verified
(loaded back and compared against PyTorch outputs). Otherwise that
check is skipped with a notice.
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


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    *,
    num_blocks: int = 4,
    num_filters: int = 64,
    value_head_hidden: int = 64,
    opset_version: int = 17,
    verify: bool = True,
) -> None:
    """Export the network to ONNX with named I/O and dynamic batch axis.

    Inputs/outputs are named so consumers can reference them by string:

    - Input  ``board``         — ``(B, 45, 9, 9)``
    - Output ``policy_logits`` — ``(B, 139, 9, 9)``
    - Output ``value``         — ``(B, 1)``

    The batch dimension is marked dynamic, so the same .onnx file
    handles single-position evaluation on a phone and large-batch
    evaluation on a server.

    Opset 17 (default) is supported by ONNX Runtime ≥ 1.13 and is a
    safe baseline; bump it if you need newer ops.
    """
    net = AlphaZeroNetwork(
        num_blocks=num_blocks,
        num_filters=num_filters,
        value_head_hidden=value_head_hidden,
    )
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # The legacy exporter pulls ``onnx`` lazily only when emitting the
    # final protobuf, producing a confusing error deep inside torch.
    # Fail early with an actionable message instead.
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ONNX export requires the `onnx` package. Install it with "
            "`pip install -e \".[export]\"` (also pulls onnxruntime for "
            "round-trip verification) or `pip install onnx`."
        ) from exc

    # eval() mode is critical here: BatchNorm switches from "use this
    # batch's stats" to "use the running stats accumulated during
    # training". Without it the exported model would be sensitive to
    # the inference batch size.
    dummy = torch.randn(1, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)

    # ``dynamo=False`` selects the legacy TorchScript-based ONNX exporter.
    # Recent torch defaults to a new dynamo-based path that requires the
    # extra ``onnxscript`` package and produces graphs that some downstream
    # toolchains (Core ML converter, older ONNX Runtime versions) do not
    # yet handle cleanly. The legacy path only needs ``onnx``, has been
    # the standard for years, and is the safer default for mobile/web
    # deployment. PyTorch 2.9+ emits a DeprecationWarning urging
    # migration; silence it locally since the trade-off is intentional.
    with warnings.catch_warnings():
        # The warning is raised by torch with the call site as its
        # ``module`` attribute, so a module=... filter does not match.
        # Match by message text instead.
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*legacy TorchScript-based ONNX export.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*setup_onnx_logging.*|.*The feature will be removed.*",
        )
        torch.onnx.export(
            net,
            dummy,
            output_path,
            input_names=["board"],
            output_names=["policy_logits", "value"],
            dynamic_axes={
                "board": {0: "batch"},
                "policy_logits": {0: "batch"},
                "value": {0: "batch"},
            },
            opset_version=opset_version,
            dynamo=False,
        )

    size_kb = os.path.getsize(output_path) / 1024
    logger.info("Exported ONNX: %s (%.1f KB)", output_path, size_kb)
    logger.info(
        "  Input:  board (B, %d, %d, %d)", NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES,
    )
    logger.info("  Output: policy_logits (B, 139, 9, 9), value (B, 1)")
    logger.info("  Opset:  %d", opset_version)

    if verify:
        _verify_round_trip(output_path, net, dummy)


def _verify_round_trip(
    onnx_path: str, net: AlphaZeroNetwork, dummy: torch.Tensor,
) -> None:
    """Load the ONNX file back and compare outputs against PyTorch.

    Skipped (with an informational log) when ``onnxruntime`` is not
    installed — the export itself only needs ``torch``, so the
    runtime is an optional verification dependency.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        logger.info(
            "  (round-trip verification skipped — `pip install onnxruntime` to enable)",
        )
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_outs = sess.run(None, {"board": dummy.numpy()})

    with torch.no_grad():
        ref_policy, ref_value = net(dummy)

    # 1e-4 tolerance: ONNX Runtime and PyTorch use slightly different
    # convolution kernels and floating-point reduction orders, so exact
    # equality is not expected. 1e-4 catches genuine bugs while
    # tolerating those numerics.
    if not np.allclose(onnx_outs[0], ref_policy.numpy(), atol=1e-4):
        raise RuntimeError("ONNX policy output diverged from PyTorch")
    if not np.allclose(onnx_outs[1], ref_value.numpy(), atol=1e-4):
        raise RuntimeError("ONNX value output diverged from PyTorch")
    logger.info("  Round-trip verified: ONNX outputs match PyTorch (atol=1e-4).")


def main():
    parser = argparse.ArgumentParser(
        description="Export an AlphaZero/Bonanza checkpoint to ONNX.",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a .pt checkpoint produced by AlphaZeroTrainer or BonanzaTrainer.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the ONNX file (.onnx).",
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
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17, requires ONNX Runtime >= 1.13).",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip the round-trip verification step.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    export_onnx(
        args.checkpoint, args.output,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        value_head_hidden=args.value_head_hidden,
        opset_version=args.opset,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
