# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shogi Dojo is the repository for the Computer Shogi Studies Research Group. It contains resources for studying and experimenting with NNUE (Efficiently Updatable Neural Network) evaluation in Shogi, including a HalfKAv2 training implementation (~5M parameters) built with PyTorch. The team is junior in computer Shogi — code and comments should be thorough and beginner-friendly.

## Commands

```bash
# Setup
source virtualenv/bin/activate
pip install -e .

# Training
python train.py                                    # default config
python train.py --config configs/my_config.yaml    # custom config
python train.py --resume checkpoints/epoch_0050.pt # resume from checkpoint

# Build YaneuraOu for training data generation (must use evallearn target + g++)
cd vendor/YaneuraOu/source && make clean && make evallearn COMPILER=g++ && cd ../../..

# No tests or linting configured yet (pytest and ruff are dev dependencies)
```

## Architecture

Data flows linearly: **constants → features → dataset → model → trainer → train.py**

- `constants.py` — Shogi piece types, board geometry, and HalfKAv2 feature dimensions. Defines the 19,638-dimensional sparse input space (9 king buckets x 2,182 features per perspective).
- `features.py` — SFEN text parsing and sparse feature index computation. Produces two sets of ~40 active indices (out of 19,638) per position: one per perspective.
- `dataset.py` — Streaming `IterableDataset` for large SFEN files. Custom `collate_fn` pads variable-length index tensors.
- `model.py` — `FeatureTransformer` (sparse lookup + sum, ~5M params) and `NnueNetwork` (3-layer eval net, ~18K params). Uses ClippedReLU throughout.
- `trainer.py` — Training loop with dual loss (score + result), best-model tracking, early stopping, LR scheduling, gradient clipping, and checkpointing.
- `train.py` — CLI entry point.
- `configs/default.yaml` — All hyperparameters. Fully commented.
- `vendor/YaneuraOu/` — Git submodule of the YaneuraOu Shogi engine, used for training data generation via its `gensfen` command. Not needed once you have `.sfen` files.

## Git Workflow

- **Do not commit directly to the `main` branch.** All work should be done on feature branches.
- Create a pull request for every change and request review from the professor ("The Boss").
- Changes may only be merged into `main` after the professor's approval.

## Key Design Concepts

**Sparse two-perspective evaluation:** Each position is encoded from both the side-to-move (STM) and not-side-to-move (NSTM) perspectives. White's perspective flips the board 180 degrees and swaps piece colors. Both perspectives share the same FeatureTransformer weights but produce separate 256-dim outputs, which are concatenated into a 512-dim vector before the eval network.

**-1 padding convention:** Positions have varying numbers of pieces, so sparse index tensors have varying lengths. The custom `collate_fn` in `dataset.py` pads shorter tensors with -1. The `FeatureTransformer.forward_dense()` in `model.py` masks these out (clamps to 0 for indexing, then zeros the result). This convention is load-bearing — both files must stay in sync.

**King buckets:** Instead of per-square feature weights for all 81 king positions (which would be enormous), king squares are grouped into 9 buckets by rank. The bucket selects which slice of the weight table to use: `index = bucket * 2182 + feature_index`. Only one bucket is active per position.

**Dual loss function:** Training optimizes a weighted sum of sigmoid-MSE against the engine score (`lambda_score=1.0`) and sigmoid-MSE against the game result (`lambda_result=0.5`). Score loss teaches tactical accuracy; result loss teaches strategic understanding.

**ClippedReLU, not ReLU:** Activations are clamped to [0, 1] for compatibility with integer quantization in game engines. This is standard for HalfKAv2 NNUE.
