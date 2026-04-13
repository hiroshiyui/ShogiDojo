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

### ~~Data Validation Utilities~~ — Done

`scripts/validate_sfen.py` validates SFEN training data files: checks for
malformed lines, reports score distribution (histogram), game result balance
(W/D/L), and optionally detects duplicate positions. Supports multiple files
with aggregate summary.

### ~~NNUE Quantization Export~~ — Done

`scripts/export_nnue_quantized.py` exports int16 (FT, scale=127) and int8
(hidden layers, scale=64) quantized weights following the Stockfish NNUE
convention. Produces ~2x smaller files than the float32 export.

### ~~Experiment Config Variants~~ — Done

- `configs/small.yaml` — ~640K params (ft_out_dim=64, l1/l2=16), 50 epochs
- `configs/large.yaml` — ~20M params (ft_out_dim=512, l1/l2=64), 400 epochs

## Nice to Have

### ~~Training Visualization~~ — Done

TensorBoard logging integrated into `trainer.py`. Logs train loss, val loss,
and learning rate per epoch. Enabled automatically when `log_dir` is set in
config. Install with `pip install -e ".[viz]"`. View with
`tensorboard --logdir runs`.

### ~~Benchmarking Suite~~ — Done

`scripts/benchmark.py` measures inference speed (positions/sec at various
batch sizes), training throughput (samples/sec), model parameter count, and
GPU memory usage. Configurable via `--config` and `--device`.

---

## Completed

### Rust gensfen Tool (`tools/gensfen/`) — Done

Standalone Rust tool replacing YaneuraOu for NNUE training data generation.
Features: alpha-beta search with iterative deepening, three-phase eval
(material -> positional -> NNUE), self-play with opening randomization,
multi-threaded generation (rayon), and text SFEN output directly consumable
by `dataset.py`.
