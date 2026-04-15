# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shogi Dojo is the repository for the Computer Shogi Studies Research Group. It contains resources for studying and experimenting with NNUE (Efficiently Updatable Neural Network) evaluation in Shogi, including a HalfKAv2 training implementation (~5M parameters) built with PyTorch. The team is junior in computer Shogi — code and comments should be thorough and beginner-friendly.

## Repository Layout

The repo is split by evaluation paradigm. NNUE assets live under `nnue/`
subdirectories everywhere (`src/shogi_dojo/nnue/`, `configs/nnue/`,
`scripts/nnue/`, `tests/nnue/`, `doc/nnue/`, `tools/nnue/gensfen/`).

A parallel **AlphaZero research survey** is being built up under matching
`alphazero/` subdirectories (`src/shogi_dojo/alphazero/`,
`doc/alphazero/`, `configs/alphazero/`, `tests/alphazero/`,
`scripts/alphazero/`, `tools/alphazero/`). The survey is educational —
docs lead, code follows, no competitive engine is targeted. See
[`doc/alphazero/AlphaZero_Reference.md`](doc/alphazero/AlphaZero_Reference.md)
for the foundational document and `doc/TODO.md` for the phased roadmap.

Top-level entry points are paradigm-prefixed (`train_nnue.py`, future
`train_alphazero.py`).

## Commands

```bash
# Setup
source virtualenv/bin/activate
pip install -e .

# Training
python train_nnue.py                                       # default config
python train_nnue.py --config configs/nnue/my_config.yaml  # custom config
python train_nnue.py --resume checkpoints/epoch_0050.pt    # resume from checkpoint

# Build the Rust gensfen tool for training data generation
cd tools/nnue/gensfen && cargo build --release && cd ../../..

# Generate training data (self-play)
./tools/nnue/gensfen/target/release/gensfen --depth 4 --threads 4 --positions 100000 --output data/train.sfen

# Tests and linting
pip install -e ".[dev]"
pytest tests/
ruff check .
```

## Architecture

Data flows linearly inside the NNUE package: **constants → features → dataset → model → trainer → train_nnue.py**

- `src/shogi_dojo/nnue/constants.py` — Shogi piece types, board geometry, and HalfKAv2 feature dimensions. Defines the 19,638-dimensional sparse input space (9 king buckets x 2,182 features per perspective).
- `src/shogi_dojo/nnue/features.py` — SFEN text parsing and sparse feature index computation. Produces two sets of ~40 active indices (out of 19,638) per position: one per perspective.
- `src/shogi_dojo/nnue/dataset.py` — Streaming `IterableDataset` for large SFEN files. Custom `collate_fn` pads variable-length index tensors.
- `src/shogi_dojo/nnue/model.py` — `FeatureTransformer` (sparse lookup + sum, ~5M params) and `NnueNetwork` (3-layer eval net, ~18K params). Uses ClippedReLU throughout.
- `src/shogi_dojo/nnue/trainer.py` — Training loop with dual loss (score + result), best-model tracking, early stopping, LR scheduling, gradient clipping, and checkpointing.
- `train_nnue.py` — CLI entry point at the repository root.
- `configs/nnue/default.yaml` — All hyperparameters. Fully commented.
- `tools/nnue/gensfen/` — Rust tool for training data generation via self-play. Uses shogi-rs for board/move generation, alpha-beta search with material eval, outputs text SFEN directly.

## Testing Policy

- **Every code addition or change must include corresponding tests.** If you add or modify functionality, add or update tests to cover the changes. Tests and implementation must stay aligned — never leave untested code behind.

## Git Workflow

- Commit directly to `main` for routine changes.
- Use feature branches and pull requests when collaboration or review is needed.

## Key Design Concepts

**Sparse two-perspective evaluation:** Each position is encoded from both the side-to-move (STM) and not-side-to-move (NSTM) perspectives. White's perspective flips the board 180 degrees and swaps piece colors. Both perspectives share the same FeatureTransformer weights but produce separate 256-dim outputs, which are concatenated into a 512-dim vector before the eval network.

**-1 padding convention:** Positions have varying numbers of pieces, so sparse index tensors have varying lengths. The custom `collate_fn` in `dataset.py` pads shorter tensors with -1. The `FeatureTransformer.forward_dense()` in `model.py` masks these out (clamps to 0 for indexing, then zeros the result). This convention is load-bearing — both files must stay in sync.

**King buckets:** Instead of per-square feature weights for all 81 king positions (which would be enormous), king squares are grouped into 9 buckets by rank. The bucket selects which slice of the weight table to use: `index = bucket * 2182 + feature_index`. Only one bucket is active per position.

**Dual loss function:** Training optimizes a weighted sum of sigmoid-MSE against the engine score (`lambda_score=1.0`) and sigmoid-MSE against the game result (`lambda_result=0.5`). Score loss teaches tactical accuracy; result loss teaches strategic understanding.

**ClippedReLU, not ReLU:** Activations are clamped to [0, 1] for compatibility with integer quantization in game engines. This is standard for HalfKAv2 NNUE.
