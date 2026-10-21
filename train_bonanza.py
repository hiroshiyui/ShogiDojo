#!/usr/bin/env python3
"""Entry point for Bonanza-method supervised training.

Usage:
    # Train on a small set of CSA files passed directly on the command line:
    python train_bonanza.py --data games.csa other.csa

    # For large datasets (e.g. the shogi-server wdoor archives contain
    # ~167K files which would use ~200 GB loaded into RAM), use both
    # --data-list (to avoid the argv limit) and --streaming (to avoid
    # OOM):
    find data/wdoor -name '*.csa' > data/wdoor_files.txt
    python train_bonanza.py --data-list data/wdoor_files.txt \
        --streaming --min-ply 10

    # Skip the first 10 plies (opening book moves) and use the default config:
    python train_bonanza.py --config configs/bonanza/default.yaml \
        --data-list data/wdoor_files.txt --streaming --min-ply 10

    # Resume training from a saved checkpoint (epoch- or step-based):
    python train_bonanza.py --resume checkpoints/bonanza_epoch_0010.pt \
        --data-list data/wdoor_files.txt --streaming

    # Fastest: preprocess the CSA files once, then train from the packed
    # shards (no CSA parsing or legal-move generation during training):
    python scripts/bonanza/preprocess_kifu.py \
        --data-list data/wdoor_files.txt --output data/packed/wdoor \
        --min-ply 10 --workers 10
    python train_bonanza.py --config configs/bonanza/default.yaml \
        --packed data/packed/wdoor

    # v2 recipe: bigger net, AMP, cosine LR, and 2% of shards (whole
    # games) held out for validation; the best model is saved as
    # checkpoints/bonanza_v2/bonanza_best.pt:
    python train_bonanza.py --config configs/bonanza/v2.yaml \
        --packed data/packed/wdoor_all --val-fraction 0.02

Before running, you need Shogi game records in CSA format.
See doc/bonanza/Bonanza_Reference.md §5 for data sources.
"""

import argparse
import logging
from pathlib import Path

import yaml

from shogi_dojo.bonanza.dataset import (
    StreamingSupervisedDataset,
    SupervisedDataset,
)
from shogi_dojo.bonanza.packed import PackedDataset
from shogi_dojo.bonanza.trainer import BonanzaTrainer


def _read_data_list(path: str) -> list[str]:
    """Read a newline-separated list of CSA file paths."""
    lines = Path(path).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def main():
    parser = argparse.ArgumentParser(description="ShogiDojo Bonanza Trainer")
    parser.add_argument(
        "--config", type=str, default="configs/bonanza/tiny.yaml",
        help="Path to training config YAML (default: configs/bonanza/tiny.yaml)",
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data", type=str, nargs="+",
        help="One or more CSA game record files. Use for small datasets; "
             "for large datasets (thousands of files) prefer --data-list.",
    )
    data_group.add_argument(
        "--data-list", type=str,
        help="Path to a text file listing one CSA file path per line. "
             "Use this for large datasets (e.g. ~167K files from the "
             "shogi-server wdoor archives) to avoid the argv limit. "
             "Build with: find data/wdoor -name '*.csa' > data/wdoor_files.txt",
    )
    data_group.add_argument(
        "--packed", type=str,
        help="Directory of pre-processed shards written by "
             "scripts/bonanza/preprocess_kifu.py. Much faster than "
             "--streaming because the CSA replay and legal-move "
             "generation were done once up front. --min-ply/--max-ply "
             "are ignored (they were applied at preprocessing time).",
    )
    parser.add_argument(
        "--min-ply", type=int, default=1,
        help="Skip positions before this ply number (default: 1 = include all).",
    )
    parser.add_argument(
        "--max-ply", type=int, default=512,
        help="Stop replaying after this ply (default: 512).",
    )
    parser.add_argument(
        "--streaming", action="store_true",
        help="Use StreamingSupervisedDataset (bounded memory, recommended "
             "for the full 167K-file wdoor dataset). Default uses the "
             "map-style SupervisedDataset which loads everything into RAM.",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=10000,
        help="Shuffle buffer size for streaming mode (default: 10000).",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint file (.pt) to resume training from. "
             "Continues from the next epoch after the saved one. If "
             "resuming from a step-based checkpoint, the remainder of "
             "that epoch is skipped.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=None,
        help="With --packed: hold out this fraction of shards (whole games) "
             "for validation, e.g. 0.02. Validation runs after every epoch "
             "and the best model is saved as bonanza_best.pt.",
    )
    parser.add_argument(
        "--val-seed", type=int, default=0,
        help="Seed for choosing the validation shards (default: 0). Keep it "
             "fixed when resuming so the split does not change.",
    )
    args = parser.parse_args()
    if args.val_fraction is not None and not args.packed:
        parser.error("--val-fraction requires --packed.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.data_list:
        data_files = _read_data_list(args.data_list)
        logging.info("Loaded %d CSA file path(s) from %s", len(data_files), args.data_list)
    else:
        data_files = args.data

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    val_dataset = None

    if args.packed:
        dataset = PackedDataset(args.packed)
        logging.info(
            "Dataset: %d packed samples in %d shard(s) from %s "
            "(min_ply=%d, max_ply=%d at preprocessing time).",
            len(dataset), dataset.num_shards, args.packed,
            dataset.config["min_ply"], dataset.config["max_ply"],
        )
        if args.val_fraction is not None:
            dataset, val_dataset = dataset.split(args.val_fraction, seed=args.val_seed)
            logging.info(
                "Validation split: %d shard(s) / %d samples held out, "
                "%d shard(s) / %d samples for training (seed=%d).",
                val_dataset.num_shards, len(val_dataset),
                dataset.num_shards, len(dataset), args.val_seed,
            )
    elif args.streaming:
        dataset = StreamingSupervisedDataset(
            csa_files=data_files,
            min_ply=args.min_ply,
            max_ply=args.max_ply,
            buffer_size=args.buffer_size,
        )
        logging.info(
            "Dataset: streaming over %d file(s), buffer_size=%d.",
            len(data_files), args.buffer_size,
        )
    else:
        dataset = SupervisedDataset.from_csa_files(
            data_files,
            min_ply=args.min_ply,
            max_ply=args.max_ply,
        )
        logging.info(
            "Dataset: %d training samples from %d file(s).",
            len(dataset), len(data_files),
        )

    trainer = BonanzaTrainer(cfg)
    if args.resume:
        trainer.resume_from(args.resume)
    trainer.train(dataset, val_dataset=val_dataset)


if __name__ == "__main__":
    main()
