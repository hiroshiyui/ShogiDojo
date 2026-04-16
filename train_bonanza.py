#!/usr/bin/env python3
"""Entry point for Bonanza-method supervised training.

Usage:
    # Train on a CSA game file with the tiny config:
    python train_bonanza.py --data games.csa

    # Train on multiple files with the default config:
    python train_bonanza.py --config configs/bonanza/default.yaml \
        --data pro_games_2024.csa pro_games_2025.csa

    # Skip the first 10 plies (opening book moves):
    python train_bonanza.py --data games.csa --min-ply 10

Before running, you need professional game records in CSA format.
See doc/bonanza/Bonanza_Reference.md §5 for data sources.
"""

import argparse
import logging

import yaml

from shogi_dojo.bonanza.dataset import SupervisedDataset
from shogi_dojo.bonanza.trainer import BonanzaTrainer


def main():
    parser = argparse.ArgumentParser(description="ShogiDojo Bonanza Trainer")
    parser.add_argument(
        "--config", type=str, default="configs/bonanza/tiny.yaml",
        help="Path to training config YAML (default: configs/bonanza/tiny.yaml)",
    )
    parser.add_argument(
        "--data", type=str, nargs="+", required=True,
        help="One or more CSA game record files to train on.",
    )
    parser.add_argument(
        "--min-ply", type=int, default=1,
        help="Skip positions before this ply number (default: 1 = include all).",
    )
    parser.add_argument(
        "--max-ply", type=int, default=512,
        help="Stop replaying after this ply (default: 512).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset = SupervisedDataset.from_csa_files(
        args.data,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
    )
    logging.info("Dataset: %d training samples from %d file(s).", len(dataset), len(args.data))

    trainer = BonanzaTrainer(cfg)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
