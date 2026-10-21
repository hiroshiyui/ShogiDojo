# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shogi Dojo is the repository for the Computer Shogi Studies Research Group. It surveys three complementary Shogi-AI paradigms side by side:

- **NNUE (HalfKAv2)** — sparse, incrementally-updatable evaluation (~5M params) for alpha-beta search. The primary pipeline: PyTorch trainer + Rust `gensfen` data generator.
- **AlphaZero-style** — ResNet policy/value network (~336K params, survey-sized) + PUCT MCTS + self-play. Educational research survey.
- **Bonanza-method** — supervised policy + value training from professional kifu records. Reuses the AlphaZero network architecture; simplest of the three pipelines. Can warm-start AlphaZero self-play.

The team is junior in computer Shogi — code and comments should be thorough and beginner-friendly.

## Repository Layout

The repo is split by paradigm. Each of the three lives under matching
subdirectories across the tree: `src/shogi_dojo/<paradigm>/`,
`configs/<paradigm>/`, `scripts/<paradigm>/`, `tests/<paradigm>/`, and
`doc/<paradigm>/`, where `<paradigm>` is one of `nnue`, `alphazero`,
or `bonanza`. NNUE-specific Rust tooling lives under
`tools/nnue/gensfen/`.

Top-level entry points are paradigm-prefixed: `train_nnue.py`,
`train_alphazero.py`, `train_bonanza.py`.

See `doc/TODO.md` for the roadmap (completed phases and open research
directions) and the paradigm-specific reference docs at
`doc/<paradigm>/*.md` for conceptual background.

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

### NNUE (`src/shogi_dojo/nnue/`)

Data flows linearly: **constants → features → dataset → model → trainer → train_nnue.py**

- `constants.py` — Shogi piece types, board geometry, and HalfKAv2 feature dimensions. Defines the 19,638-dimensional sparse input space (9 king buckets x 2,182 features per perspective).
- `features.py` — SFEN text parsing and sparse feature index computation. Produces two sets of ~40 active indices (out of 19,638) per position: one per perspective.
- `dataset.py` — Streaming `IterableDataset` for large SFEN files. Custom `collate_fn` pads variable-length index tensors.
- `model.py` — `FeatureTransformer` (sparse lookup + sum, ~5M params) and `NnueNetwork` (3-layer eval net, ~18K params). Uses ClippedReLU throughout.
- `trainer.py` — Training loop with dual loss (score + result), best-model tracking, early stopping, LR scheduling, gradient clipping, and checkpointing.
- `train_nnue.py` — CLI entry point at the repository root.
- `configs/nnue/default.yaml` — All hyperparameters. Fully commented.
- `tools/nnue/gensfen/` — Rust tool for training data generation via self-play. Uses shogi-rs for board/move generation, alpha-beta search with material eval, outputs text SFEN directly.

### AlphaZero survey (`src/shogi_dojo/alphazero/`)

Self-play training loop: **encoding → network → mcts → rules → self_play → trainer → train_alphazero.py**

- `encoding.py` — Board → 45-channel input tensor; Move ↔ flat 9×9×139 policy index (paper-faithful action space). `position_from_sfen` bridges the NNUE SFEN parser.
- `network.py` — `AlphaZeroNetwork`: ResNet with policy head (139 logit planes) and value head (tanh scalar). ~336K params by default. Helpers: `masked_log_softmax`, `alphazero_loss`.
- `mcts.py` — Game-agnostic PUCT MCTS (`Node`, `Edge`, `select_child`, Dirichlet noise, search loop). Takes four callables; no Shogi-specific code.
- `rules.py` — python-shogi adapter: `legal_moves_fn`, `apply_fn`, `is_terminal_fn`, `make_evaluate_fn`. Piece-type translation (python-shogi uses 1-indexed IDs).
- `self_play.py` — `play_game()` runs one game with MCTS at every move, records (input, π, legal_mask), backfills terminal z. `ReplayBuffer` for training sampling.
- `evaluation.py` — `run_match()` and `MatchResult`: color-swapped N-game matches between two AlphaZero-family networks, with Elo-diff estimate. Used for promotion gating and cross-paradigm comparison. CLI at `scripts/alphazero/match.py`.
- `scripts/alphazero/export_torchscript.py` and `export_onnx.py` — convert any AlphaZero-architecture checkpoint (AlphaZero- or Bonanza-trained) to TorchScript (`.ts`) or ONNX (`.onnx`) for deployment outside the PyTorch Python runtime (libtorch C++, ONNX Runtime, Core ML, web). ONNX export requires the `[export]` extra (`onnx`); TorchScript needs no extra deps.
- `scripts/alphazero/make_random_checkpoint.py` — save a freshly-initialized AlphaZero-architecture checkpoint with the same payload layout as trainer-produced checkpoints. Primary use: a sanity-floor opponent for `match.py` to confirm a trained network actually plays better than random initialization.
- `trainer.py` — `AlphaZeroTrainer`: config-driven iteration loop (self-play → buffer → gradient steps → checkpoint). Supports `model.warm_start` config key for initializing from a Bonanza (or prior AlphaZero) checkpoint.
- `train_alphazero.py` — CLI entry point at the repository root.
- `configs/alphazero/{tiny,default}.yaml` — Fully commented configurations.
- `doc/alphazero/` — Four reference documents: architecture, training methodology, NNUE comparison, annotated reading list.

### Bonanza-method survey (`src/shogi_dojo/bonanza/`)

Supervised policy + value learning from professional game records (kifu). Reuses `AlphaZeroNetwork` without modification; only the training data source and loss formulation differ. State dicts are interchangeable — a Bonanza-trained checkpoint can warm-start AlphaZero self-play directly.

- `dataset.py` — CSA-format parser (via python-shogi), game replay producing `(input, target_move_index, legal_mask, value_target)` samples. Two PyTorch dataset classes: `SupervisedDataset` (map-style, loads all samples into memory) and `StreamingSupervisedDataset` (`IterableDataset` with sliding-buffer shuffle, bounded memory for the full ~167K-file wdoor archive).
- `packed.py` — one-time preprocessing of CSA kifu into compact memory-mapped NumPy shards (bit-packed piece planes, per-plane scalars, ragged legal-move index lists; ~510 B/position). `preprocess_csa_files` (multi-process, resumable, shard-atomic) and `PackedDataset` (map-style, batch-decoded via `__getitems__` + identity `collate_fn`). Removes python-shogi from the training loop (~2.2K → ~90K samples/s loader throughput). CLI at `scripts/bonanza/preprocess_kifu.py`; train with `train_bonanza.py --packed DIR`.
- `trainer.py` — `BonanzaTrainer`: epoch-based supervised loop with cross-entropy on the policy head and MSE on the value head. Auto-detects map-style vs streaming datasets and configures the DataLoader accordingly. Optional (off by default, v1-compatible): `amp` (float16 autocast + GradScaler; losses stay float32 because of the `-1e9` mask fill), `optimizer` (adam/adamw/sgd), per-step `lr_schedule` (warmup + constant/cosine/step via `make_lr_lambda`), `log_every_steps`, and `train(dataset, val_dataset=...)` which evaluates each epoch and keeps `bonanza_best.pt`. `resume_from(path)` restores model + optimizer (+ scheduler, scaler, best_val_loss when present) state and sets `start_epoch` / `global_step` so training continues seamlessly from epoch- or step-based checkpoints.
- `train_bonanza.py` — CLI entry point. Supports `--data`, `--data-list` (for large file sets), `--streaming`, `--packed` (pre-processed shards), `--val-fraction`/`--val-seed` (game-level validation split of packed shards via `PackedDataset.split`), and `--resume` (continue from a saved checkpoint) flags.
- `configs/bonanza/{tiny,default,v2}.yaml` — Fully commented configurations. `default` is the v1 recipe (4×64, Adam, constant LR); `v2` is 10×128 + AMP + AdamW + cosine LR for the packed full wdoor archive.
- `doc/bonanza/Bonanza_Reference.md` — Reference document covering history, formulation, CSA format, data sources (wdoor archive), and three-paradigm comparison.

## Testing Policy

- **Every code addition or change must include corresponding tests.** If you add or modify functionality, add or update tests to cover the changes. Tests and implementation must stay aligned — never leave untested code behind.

## Git Workflow

- Commit directly to `main` for routine changes.
- Use feature branches and pull requests when collaboration or review is needed.

## Key Design Concepts

### NNUE (paradigm-specific)

**Sparse two-perspective evaluation:** Each position is encoded from both the side-to-move (STM) and not-side-to-move (NSTM) perspectives. White's perspective flips the board 180 degrees and swaps piece colors. Both perspectives share the same FeatureTransformer weights but produce separate 256-dim outputs, which are concatenated into a 512-dim vector before the eval network.

**-1 padding convention:** Positions have varying numbers of pieces, so sparse index tensors have varying lengths. The custom `collate_fn` in `src/shogi_dojo/nnue/dataset.py` pads shorter tensors with -1. `FeatureTransformer.forward_dense()` in `src/shogi_dojo/nnue/model.py` masks these out (clamps to 0 for indexing, then zeros the result). This convention is load-bearing — both files must stay in sync.

**King buckets:** Instead of per-square feature weights for all 81 king positions (which would be enormous), king squares are grouped into 9 buckets by rank. The bucket selects which slice of the weight table to use: `index = bucket * 2182 + feature_index`. Only one bucket is active per position. Total input dimension = 9 × 2,182 = **19,638**.

**Dual loss function:** Training optimizes a weighted sum of sigmoid-MSE against the engine score (`lambda_score=1.0`) and sigmoid-MSE against the game result (`lambda_result=0.5`). Score loss teaches tactical accuracy; result loss teaches strategic understanding.

**ClippedReLU, not ReLU:** Activations are clamped to [0, 1] for compatibility with integer quantization in game engines. This is standard for HalfKAv2 NNUE.

### AlphaZero (paradigm-specific)

**Perspective sign-flip on MCTS backup:** Terminal values and network value-head outputs are always from the STM's perspective. When backing a leaf value up the search path, the sign flips at every ply because the perspective alternates. Implemented in `src/shogi_dojo/alphazero/mcts.py::MCTS._simulate`. Getting this wrong produces a network that consistently prefers losing — hence the explicit test in `test_mcts.py`.

**Dense 9×9×139 action space:** Policy output is `(B, 139, 9, 9)` flattened to 11,259 logits per position. The 139 planes decompose as 64 queen-like sliding moves + 2 knight offsets + 66 promoting variants + 7 drops, matching the AlphaZero paper's Shogi encoding. See `src/shogi_dojo/alphazero/encoding.py` for the layout.

**Legal-move masking via `-1e9` fill:** Before softmax, illegal logits are set to a large negative number (`masked_log_softmax` in `network.py`). This keeps illegal actions at ≈ 0 probability and 0 gradient without requiring a custom softmax.

### Cross-paradigm (architectural sharing)

**AlphaZero and Bonanza share the network.** Both use `AlphaZeroNetwork` with no modification. The only difference is what the heads are trained against:

| Head  | AlphaZero target                  | Bonanza target                          |
|-------|-----------------------------------|-----------------------------------------|
| Policy | MCTS visit distribution (π)       | Professional's move (one-hot)           |
| Value  | Terminal outcome z (from self-play) | Game outcome from this position's STM  |

Because the architectures are byte-for-byte identical, state dicts are interchangeable: `AlphaZeroTrainer` accepts a `model.warm_start` config key pointing at a Bonanza checkpoint, and the load is a plain `load_state_dict` with no conversion layer. The Bonanza paradigm is thus a "free" warm-start generator for AlphaZero self-play.
