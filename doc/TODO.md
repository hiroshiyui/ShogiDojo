# TODO

## Open — Shogi rule coverage gaps

The 2026-04-14 rule-compliance audit surfaced three gaps that remain.
None blocks training, but fixing them would improve data fidelity:

- **Try rule (入玉宣言勝ち, "entering king")** — not detected. A king that
  reaches the opponent's promotion zone under the usual point-count
  conditions should be declared a winner; `game.rs` currently lets such
  games drift to the `MAX_PLY` draw.
- **Impasse / 持将棋 (jishōgi)** — not detected. When both kings enter
  opposing promotion zones and neither side can mate, the game is
  adjudicated by point count, not left to time out at ply 400.
- **Defense-in-depth uchi-fu-zume** — currently delegated to shogi-rs
  via `MoveError::Uchifuzume`. `game::tests::uchi_fu_zume_pawn_drop_is_filtered`
  locks in the behaviour end-to-end, but we do not have a local,
  shogi-rs-independent implementation.

---

## Completed

### Rust gensfen Tool — `tools/gensfen/`

Standalone Rust tool for NNUE training data generation via self-play.
Alpha-beta search with iterative deepening, PST and NNUE evaluation,
opening randomization, multi-threaded generation (rayon), text SFEN output.

### CI/CD Pipeline — `.github/workflows/ci.yml`

GitHub Actions on every push/PR to `main`:
Python (ruff + pytest) and Rust (fmt + clippy + test).

### End-to-End Orchestration Script — `scripts/train_loop.py`

Automates the iterative self-improvement loop:
gensfen -> train -> export -> regenerate with NNUE eval.
Configurable iterations, positions, depth, train/val split, resume support.

### Model Evaluation Tool — `eval_match` binary

Plays games between two NNUE models (or model vs. PST baseline) with
color swapping for fairness. Reports win/loss/draw and estimated Elo difference.

### Data Validation Utilities — `scripts/validate_sfen.py`

Validates SFEN training data: malformed line detection, score distribution
histogram, result balance (W/D/L), duplicate detection, multi-file summary.

### NNUE Quantization Export — `scripts/export_nnue_quantized.py`

Exports int16 (FT, scale=127) and int8 (hidden layers, scale=64) quantized
weights following the Stockfish NNUE convention. ~2x smaller than float32.

### Experiment Config Variants — `configs/small.yaml`, `configs/large.yaml`

Small (~640K params, ft_out_dim=64) for quick experiments.
Large (~20M params, ft_out_dim=512) for maximum strength.

### Training Visualization — TensorBoard in `trainer.py`

Logs train/val loss and learning rate per epoch. Enabled when `log_dir` is
set in config. Install with `pip install -e ".[viz]"`.

### Benchmarking Suite — `scripts/benchmark.py`

Measures inference speed (positions/sec), training throughput (samples/sec),
model parameter count, and GPU memory usage.
