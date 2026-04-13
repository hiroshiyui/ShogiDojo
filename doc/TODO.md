# TODO

## High Priority

### ~~CI/CD Pipeline~~ — Done

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and PR to `main`:

- **Python job**: `ruff check`, `pytest tests/ -v`
- **Rust job**: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`

### ~~End-to-End Orchestration Script~~ — Done

`scripts/train_loop.py` automates the iterative self-improvement loop:
gensfen (PST or NNUE) -> train -> export -> repeat with stronger model.
Supports configurable iterations, positions, depth, train/val split,
resume from a previous iteration, and timestamped output directories.

### ~~Model Evaluation Tool~~ — Done

`eval_match` binary plays games between two NNUE models (or model vs. PST baseline)
with automatic color swapping for fairness. Reports win/loss/draw counts, percentages,
and estimated Elo difference. Multi-threaded via rayon. Built as a second binary in
the gensfen crate, reusing the existing search/eval/movegen infrastructure via `lib.rs`.

## Medium Priority

### Data Validation Utilities

Write a script to validate SFEN training data files:

- Check for malformed lines (currently silently skipped by `dataset.py`)
- Report statistics: score distribution, result balance (win/draw/loss ratio), position count
- Detect duplicates

### NNUE Quantization Export

`scripts/export_nnue.py` currently exports float32 weights. Add int8/int16 quantized export for integration with a Shogi game engine. Reference: the [Stockfish NNUE wiki](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html) documents scaling factors (127 for feature transformer int16, 64 for hidden layer int8).

### Experiment Config Variants

Add alternative configs for different training scenarios:

- `configs/small.yaml` — smaller model for quick experiments (e.g., `ft_out_dim=64`, `l1_dim=16`)
- `configs/large.yaml` — larger model for maximum strength (e.g., `ft_out_dim=512`, `l1_dim=64`)

Currently only `configs/default.yaml` exists.

## Nice to Have

### Training Visualization

Add TensorBoard logging or plotting scripts for loss curves, learning rate schedules, and convergence analysis. The config already reserves `log_dir: "runs"` but it is unused.

### Benchmarking Suite

Measure and track inference speed (positions/second), training throughput (samples/second), and memory usage across model sizes and hardware.

---

## Completed

### Rust gensfen Tool (`tools/gensfen/`) — Done

Standalone Rust tool replacing YaneuraOu for NNUE training data generation.
Features: alpha-beta search with iterative deepening, three-phase eval
(material -> positional -> NNUE), self-play with opening randomization,
multi-threaded generation (rayon), and text SFEN output directly consumable
by `dataset.py`.
