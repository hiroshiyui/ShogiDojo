#!/usr/bin/env python3
"""Pre-process CSA kifu into packed shards for fast Bonanza training.

Replaying games and generating legal moves with python-shogi is slow
(~2,200 samples/s with 8 DataLoader workers on the v1 run), and the
streaming dataset repeats that work every epoch. This script does it
once and writes compact binary shards that ``train_bonanza.py --packed``
reads at GPU speed. See ``src/shogi_dojo/bonanza/packed.py`` for the
on-disk format.

The run is resumable: if it is interrupted, re-run the same command and
finished shards are skipped. Keep the file list (and its order) the same
when resuming, because the list order decides which shard each file
belongs to.

Usage:
    # Build the file list once, sorted so it is stable across re-runs:
    find data/wdoor -name '*.csa' | sort > data/wdoor_files.txt

    # Preprocess with 10 worker processes (leave a couple of cores free):
    python scripts/bonanza/preprocess_kifu.py \\
        --data-list data/wdoor_files.txt --output data/packed/wdoor \\
        --min-ply 10 --workers 10

    # Then train from the packed data:
    python train_bonanza.py --config configs/bonanza/default.yaml \\
        --packed data/packed/wdoor

Disk usage is roughly 550 bytes per position, i.e. ~55 MB per 1,000
wdoor games.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from shogi_dojo.bonanza.packed import preprocess_csa_files

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data", nargs="+", help="CSA files to preprocess.")
    data_group.add_argument(
        "--data-list",
        help="Text file with one CSA path per line (use for large sets).",
    )
    parser.add_argument("--output", required=True, help="Output directory for the shards.")
    parser.add_argument(
        "--min-ply", type=int, default=1,
        help="Skip positions before this ply (default: 1 = include all). "
             "Baked into the output; cannot be changed at training time.",
    )
    parser.add_argument("--max-ply", type=int, default=512,
                        help="Stop replaying after this ply (default: 512).")
    parser.add_argument(
        "--files-per-shard", type=int, default=1000,
        help="CSA files per shard (default: 1000 ≈ 100K positions ≈ 55 MB).",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
        help="Parallel worker processes (default: CPU count - 2).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.data_list:
        lines = Path(args.data_list).read_text().splitlines()
        files = [ln.strip() for ln in lines if ln.strip()]
    else:
        files = args.data
    total_shards = -(-len(files) // args.files_per_shard)  # ceil division
    logger.info(
        "Preprocessing %d file(s) into %d shard(s) at %s with %d worker(s).",
        len(files), total_shards, args.output, args.workers,
    )

    t0 = time.time()
    done = [0, 0]  # shards, samples — mutable so the callback can update it

    def progress(result: dict) -> None:
        done[0] += 1
        done[1] += result["num_samples"]
        elapsed = time.time() - t0
        logger.info(
            "shard %05d: %d samples, %d games, %d failed file(s)  "
            "[%d new shard(s) done, %.0f samples/s]",
            result["shard_id"], result["num_samples"], result["num_games"],
            result["num_failed"], done[0], done[1] / max(elapsed, 1e-9),
        )

    summary = preprocess_csa_files(
        files,
        args.output,
        files_per_shard=args.files_per_shard,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        num_workers=args.workers,
        progress=progress,
    )
    logger.info(
        "Done in %.1f min: built %d shard(s) (%d samples, %d games, "
        "%d unreadable file(s)); skipped %d already-finished shard(s).",
        (time.time() - t0) / 60, summary.shards_built, summary.samples_built,
        summary.games_built, summary.failed_files, summary.shards_skipped,
    )


if __name__ == "__main__":
    main()
