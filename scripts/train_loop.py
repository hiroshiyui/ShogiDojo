#!/usr/bin/env python3
"""Iterative NNUE training loop.

Automates the self-improvement cycle:
  1. Generate training data via self-play (gensfen)
  2. Train the NNUE model (train.py)
  3. Export the trained weights to binary format (export_nnue.py)
  4. Repeat with the stronger NNUE model driving the next round of self-play

Each iteration produces stronger training data because the NNUE evaluator
plays better than the previous iteration's evaluator. The first iteration
uses the built-in PST (material + piece-square table) evaluator since no
trained model exists yet.

Usage:
    python scripts/train_loop.py --iterations 3 --positions 100000
    python scripts/train_loop.py --iterations 5 --positions 500000 --depth 5
    python scripts/train_loop.py --resume-from 2  # resume from iteration 2

All intermediate data, checkpoints, and models are saved under a timestamped
output directory (e.g., loop_output/20260413_143000/) so that multiple runs
don't overwrite each other.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENSFEN_BIN = PROJECT_ROOT / "tools" / "gensfen" / "target" / "release" / "gensfen"
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "export_nnue.py"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the iterative NNUE training loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of training iterations to run (default: 3).",
    )
    parser.add_argument(
        "--positions", type=int, default=100_000,
        help="Positions to generate per iteration (default: 100000).",
    )
    parser.add_argument(
        "--depth", type=int, default=4,
        help="Search depth for gensfen (default: 4).",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Threads for gensfen (default: 4).",
    )
    parser.add_argument(
        "--random-moves", type=int, default=8,
        help="Random opening moves per game (default: 8).",
    )
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG),
        help="Training config YAML (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Defaults to loop_output/<timestamp>/.",
    )
    parser.add_argument(
        "--resume-from", type=int, default=None,
        help="Resume from iteration N (uses existing output-dir data).",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.05,
        help="Fraction of generated positions reserved for validation (default: 0.05).",
    )
    return parser.parse_args()


def run_command(cmd: list[str], description: str) -> None:
    """Run a shell command, streaming output to the terminal."""
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("%s failed with exit code %d", description, result.returncode)
        sys.exit(1)
    logger.info("%s completed successfully.", description)


def split_sfen(input_path: Path, train_path: Path, val_path: Path, val_ratio: float) -> None:
    """Split a generated SFEN file into training and validation sets.

    Reads the file once, writes lines to train or val based on a simple
    modular split (every Nth line goes to validation, where N = 1/val_ratio).
    """
    if val_ratio <= 0:
        shutil.copy2(input_path, train_path)
        return

    val_every = max(1, int(1.0 / val_ratio))
    train_count = 0
    val_count = 0

    with (
        open(input_path, "r") as f_in,
        open(train_path, "w") as f_train,
        open(val_path, "w") as f_val,
    ):
        for i, line in enumerate(f_in):
            if i % val_every == 0:
                f_val.write(line)
                val_count += 1
            else:
                f_train.write(line)
                train_count += 1

    logger.info(
        "Split %s: %d train, %d val (%.1f%% val)",
        input_path.name, train_count, val_count,
        val_count / max(train_count + val_count, 1) * 100,
    )


def run_gensfen(
    iteration: int,
    output_path: Path,
    args: argparse.Namespace,
    model_path: Path | None,
) -> None:
    """Generate training data via self-play."""
    cmd = [
        str(GENSFEN_BIN),
        "--depth", str(args.depth),
        "--threads", str(args.threads),
        "--positions", str(args.positions),
        "--random-moves", str(args.random_moves),
        "--output", str(output_path),
    ]
    if model_path is not None:
        cmd.extend(["--model", str(model_path)])
        logger.info("Iteration %d: generating data with NNUE model %s", iteration, model_path.name)
    else:
        logger.info("Iteration %d: generating data with PST evaluator (no model)", iteration)

    run_command(cmd, f"gensfen (iteration {iteration})")


def run_training(
    iteration: int,
    config_path: Path,
    train_data: Path,
    val_data: Path,
    checkpoint_dir: Path,
) -> Path:
    """Train the NNUE model and return the path to best_model.pt."""
    # Create a temporary config that points to this iteration's data and checkpoint dir.
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["train_path"] = str(train_data)
    cfg["data"]["val_path"] = str(val_data)
    cfg["training"]["checkpoint_dir"] = str(checkpoint_dir)

    iter_config = checkpoint_dir.parent / f"config_iter{iteration}.yaml"
    with open(iter_config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [sys.executable, str(TRAIN_SCRIPT), "--config", str(iter_config)]
    run_command(cmd, f"training (iteration {iteration})")

    best_model = checkpoint_dir / "best_model.pt"
    if not best_model.exists():
        logger.error("Training did not produce best_model.pt in %s", checkpoint_dir)
        sys.exit(1)

    return best_model


def run_export(checkpoint_path: Path, nnue_path: Path, iteration: int) -> None:
    """Export a PyTorch checkpoint to binary NNUE format."""
    cmd = [sys.executable, str(EXPORT_SCRIPT), str(checkpoint_path), str(nnue_path)]
    run_command(cmd, f"export (iteration {iteration})")


def main() -> None:
    args = parse_args()

    # Validate prerequisites.
    if not GENSFEN_BIN.exists():
        logger.error(
            "gensfen binary not found at %s. "
            "Build it first: cd tools/gensfen && cargo build --release",
            GENSFEN_BIN,
        )
        sys.exit(1)

    # Set up output directory.
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "loop_output" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("NNUE Iterative Training Loop")
    logger.info("  Iterations:  %d", args.iterations)
    logger.info("  Positions:   %d per iteration", args.positions)
    logger.info("  Depth:       %d", args.depth)
    logger.info("  Threads:     %d", args.threads)
    logger.info("  Config:      %s", args.config)
    logger.info("  Output dir:  %s", out_dir)
    logger.info("=" * 60)

    start_iter = args.resume_from if args.resume_from is not None else 0
    model_path: Path | None = None

    # If resuming, pick up the model from the previous iteration.
    if start_iter > 0:
        prev_model = out_dir / f"model_iter{start_iter - 1}.nnue"
        if prev_model.exists():
            model_path = prev_model
            logger.info("Resuming from iteration %d, using model %s", start_iter, prev_model.name)
        else:
            logger.warning(
                "Resume requested from iteration %d but %s not found. "
                "Starting without a model.",
                start_iter, prev_model,
            )

    for iteration in range(start_iter, args.iterations):
        logger.info("")
        logger.info("=" * 60)
        logger.info("ITERATION %d / %d", iteration, args.iterations - 1)
        logger.info("=" * 60)

        iter_dir = out_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        raw_sfen = iter_dir / "raw.sfen"
        train_sfen = iter_dir / "train.sfen"
        val_sfen = iter_dir / "val.sfen"
        ckpt_dir = iter_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate data.
        t0 = time.time()
        run_gensfen(iteration, raw_sfen, args, model_path)
        logger.info("Data generation took %.1f seconds.", time.time() - t0)

        # Step 2: Split into train/val.
        split_sfen(raw_sfen, train_sfen, val_sfen, args.val_ratio)

        # Step 3: Train.
        t0 = time.time()
        best_checkpoint = run_training(
            iteration, Path(args.config), train_sfen, val_sfen, ckpt_dir,
        )
        logger.info("Training took %.1f seconds.", time.time() - t0)

        # Step 4: Export to NNUE binary.
        nnue_path = out_dir / f"model_iter{iteration}.nnue"
        run_export(best_checkpoint, nnue_path, iteration)

        # The exported model becomes the evaluator for the next iteration.
        model_path = nnue_path
        logger.info("Iteration %d complete. Model saved to %s", iteration, nnue_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("All %d iterations complete.", args.iterations)
    logger.info("Final model: %s", model_path)
    logger.info("Output directory: %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
