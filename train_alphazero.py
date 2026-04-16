#!/usr/bin/env python3
"""Entry point for AlphaZero-style self-play training.

Usage:
    # Run the tiny smoke-test configuration:
    python train_alphazero.py --config configs/alphazero/tiny.yaml

    # Run the default configuration (hours-scale):
    python train_alphazero.py --config configs/alphazero/default.yaml

Before running, make sure you have python-shogi installed (it should be
part of the standard ``pip install -e .`` install since it is listed in
pyproject.toml).

See doc/alphazero/AlphaZero_Training_Reference.md for a full explanation
of the training loop, hyperparameters, and resource expectations.
"""

import argparse
import logging

import yaml

from shogi_dojo.alphazero.trainer import AlphaZeroTrainer


def main():
    parser = argparse.ArgumentParser(description="ShogiDojo AlphaZero Trainer")
    parser.add_argument(
        "--config", type=str, default="configs/alphazero/tiny.yaml",
        help="Path to training config YAML (default: configs/alphazero/tiny.yaml)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trainer = AlphaZeroTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
