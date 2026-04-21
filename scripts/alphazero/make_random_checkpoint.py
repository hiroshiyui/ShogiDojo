#!/usr/bin/env python3
"""Save a freshly-initialized AlphaZero-architecture checkpoint.

The output file is shaped exactly like a checkpoint produced by
``AlphaZeroTrainer`` or ``BonanzaTrainer`` — same payload keys, same
state-dict layout — so it can be loaded by ``scripts/alphazero/match.py``
and other consumers without special handling.

The primary use is **producing a sanity-floor opponent**: matching a
trained network against this baseline confirms that training actually
moved the network past random initialization. A trained net that does
not crush a freshly-initialized one (~95%+) is a sign that something
went wrong (architecture mismatch, weight-loading bug, untrained
checkpoint, etc.).

Usage:
    # Default architecture (matches configs/{alphazero,bonanza}/default.yaml):
    python scripts/alphazero/make_random_checkpoint.py \\
        --output checkpoints/random_init.pt

    # Reproducible (fixed seed → identical weights every run):
    python scripts/alphazero/make_random_checkpoint.py \\
        --output checkpoints/random_init.pt --seed 0

    # Match a non-default network size:
    python scripts/alphazero/make_random_checkpoint.py \\
        --output checkpoints/random_large.pt \\
        --num-blocks 8 --num-filters 128

Then play it against your trained network:

    python scripts/alphazero/match.py \\
        --network-a checkpoints/bonanza/bonanza_epoch_0050.pt \\
        --network-b checkpoints/random_init.pt --num-games 20
"""

from __future__ import annotations

import argparse
import logging
import os

import torch

from shogi_dojo.alphazero.network import AlphaZeroNetwork

logger = logging.getLogger(__name__)


def make_random_checkpoint(
    output_path: str,
    *,
    num_blocks: int = 4,
    num_filters: int = 64,
    value_head_hidden: int = 64,
    seed: int | None = None,
) -> None:
    """Build an ``AlphaZeroNetwork`` and save its initial weights.

    The payload mirrors what the trainers save, so loaders that look
    for ``model_state_dict``, ``optimizer_state_dict``, ``epoch``, and
    ``global_step`` keys (e.g. ``match.py``) work unchanged.
    ``optimizer_state_dict`` is empty since no training has happened.

    If ``seed`` is provided, ``torch.manual_seed`` is called before
    constructing the network so the random initialization is fully
    reproducible. Useful when you want every team member to evaluate
    against an identical baseline.
    """
    if seed is not None:
        torch.manual_seed(seed)

    net = AlphaZeroNetwork(
        num_blocks=num_blocks,
        num_filters=num_filters,
        value_head_hidden=value_head_hidden,
    )
    param_count = sum(p.numel() for p in net.parameters())

    torch.save(
        {
            "epoch": 0,
            "global_step": 0,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": {},
        },
        output_path,
    )

    size_kb = os.path.getsize(output_path) / 1024
    logger.info(
        "Saved random checkpoint: %s (%.1f KB, %s params)",
        output_path, size_kb, f"{param_count:,}",
    )
    logger.info(
        "  Architecture: num_blocks=%d num_filters=%d value_head_hidden=%d",
        num_blocks, num_filters, value_head_hidden,
    )
    if seed is not None:
        logger.info("  Seed: %d (re-running with this seed gives identical weights)", seed)


def main():
    parser = argparse.ArgumentParser(
        description="Save a randomly-initialized AlphaZero-architecture checkpoint.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the checkpoint (.pt).",
    )
    parser.add_argument(
        "--num-blocks", type=int, default=4,
        help="Number of residual blocks (default: 4, matches default configs).",
    )
    parser.add_argument(
        "--num-filters", type=int, default=64,
        help="Number of conv filters per block (default: 64).",
    )
    parser.add_argument(
        "--value-head-hidden", type=int, default=64,
        help="Value head hidden size (default: 64).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional RNG seed for reproducible weights. Omit for "
             "non-deterministic initialization.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    make_random_checkpoint(
        args.output,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        value_head_hidden=args.value_head_hidden,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
