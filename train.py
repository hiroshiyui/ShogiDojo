#!/usr/bin/env python3
"""Entry point for Shogi NNUE training.

Usage:
    # Start training from scratch with default config:
    python train.py

    # Use a custom config file:
    python train.py --config configs/my_config.yaml

    # Resume training from a checkpoint:
    python train.py --resume checkpoints/epoch_0050.pt

Before running, make sure your training data is placed in the data/ directory
(or wherever your config points to). See dataset.py for the expected format.
"""

import argparse
import logging
import yaml

from shogi_dojo.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="ShogiDojo NNUE Trainer")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to training config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint file (.pt) to resume training from",
    )
    args = parser.parse_args()

    # Set up logging so we can see training progress in the terminal
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load the training configuration from YAML
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Create the trainer (builds model, optimizer, scheduler)
    trainer = Trainer(cfg)

    # If resuming from a checkpoint, load the saved model and optimizer state.
    # This restores the training to exactly where it left off, including
    # the best-model tracking state so early stopping works correctly.
    if args.resume:
        import torch
        ckpt = torch.load(args.resume, weights_only=True)
        trainer.model.load_state_dict(ckpt['model_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Restore best-model tracking state (if saved in the checkpoint)
        if 'best_loss' in ckpt:
            trainer.best_loss = ckpt['best_loss']
            trainer.best_epoch = ckpt['best_epoch']
        logging.info(
            f"Resumed from {args.resume} (epoch {ckpt['epoch']}, "
            f"best_loss={trainer.best_loss:.6f} at epoch {trainer.best_epoch})"
        )

    # Start training!
    trainer.train()


if __name__ == "__main__":
    main()
