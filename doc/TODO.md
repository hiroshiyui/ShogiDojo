# TODO

## Open — AlphaZero Survey

Research survey of the AlphaZero family of game-playing systems, run in
parallel with the NNUE work and mirrored under `alphazero/` subdirectories
(`src/shogi_dojo/alphazero/`, `doc/alphazero/`, `configs/alphazero/`,
`tests/alphazero/`, `scripts/alphazero/`, `tools/alphazero/`). The survey
is *educational* — it does not aim to produce a competitive engine. The
team is junior in computer Shogi, so docs lead and code follows.

Phased plan:

- **Phase 2.1 — Skeleton** *(done)*: `alphazero/` subdirs in place with
  `__init__.py` for the new Python sub-package.
- **Phase 2.2 — Foundational doc** *(done)*: `doc/alphazero/AlphaZero_Reference.md`
  covers history, architecture (ResNet + policy/value heads), Shogi input
  and move encoding, MCTS+PUCT, the self-play loop, the loss function,
  comparison with NNUE, glossary, and references.
- **Phase 2.3 — Training reference** *(done)*:
  `doc/alphazero/AlphaZero_Training_Reference.md` mirrors
  `doc/nnue/Model_Training_Reference.md` — self-play methodology,
  replay-buffer dynamics, loss decomposition, hyperparameter ranges
  (Dirichlet noise, temperature schedule, c_puct, MCTS simulations),
  the training loop step by step, evaluation/promotion gating,
  distributed self-play architecture, resource estimates, common
  failure modes, improvement strategy, side-by-side comparison with
  NNUE training, and realistic expectations for the survey.
- **Phase 2.4 — Encoding code + tests** *(planned)*:
  `src/shogi_dojo/alphazero/encoding.py` (board → input planes,
  move ↔ policy index) with round-trip tests under
  `tests/alphazero/`. Reuse `shogi-rs` bindings for legality where
  possible rather than reimplementing rules.
- **Phase 2.5 — Network code + tests** *(planned)*:
  `src/shogi_dojo/alphazero/network.py` — small ResNet with policy and
  value heads, sized for laptop training (configurable via
  `configs/alphazero/tiny.yaml`).
- **Phase 2.6 — MCTS code + tests** *(planned)*:
  `src/shogi_dojo/alphazero/mcts.py` — readable PUCT MCTS over fast.
  Tests cover invariants (visit-sum conservation, perspective sign-flip
  on backup, Dirichlet noise application).
- **Phase 2.7 — Self-play + trainer + smoke test** *(planned)*:
  `self_play.py`, `trainer.py`, `scripts/alphazero/selfplay.py`,
  `scripts/alphazero/train.py` (top-level entry point becomes
  `train_alphazero.py`), `configs/alphazero/{tiny,default}.yaml`, and an
  end-to-end smoke test.
- **Phase 2.8 — Comparison and reading list** *(planned)*:
  `doc/alphazero/AlphaZero_vs_NNUE.md` (operational comparison with
  numbers from §10 of the reference) and `doc/alphazero/AlphaZero_Reading_List.md`
  (annotated bibliography expanded from §12).
- **Phase 2.9 — README integration** *(planned)*: an "AlphaZero Survey"
  section in `README.md` parallel to the existing NNUE section, and an
  `alphazero/` subsection under Architecture in `CLAUDE.md`.

Sequencing is loose: docs and code can interleave (writing
`encoding.py` may surface clarifications worth folding back into the
reference doc, and vice versa).

---

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

### Rust gensfen Tool — `tools/nnue/gensfen/`

Standalone Rust tool for NNUE training data generation via self-play.
Alpha-beta search with iterative deepening, PST and NNUE evaluation,
opening randomization, multi-threaded generation (rayon), text SFEN output.

### CI/CD Pipeline — `.github/workflows/ci.yml`

GitHub Actions on every push/PR to `main`:
Python (ruff + pytest) and Rust (fmt + clippy + test).

### End-to-End Orchestration Script — `scripts/nnue/train_loop.py`

Automates the iterative self-improvement loop:
gensfen -> train -> export -> regenerate with NNUE eval.
Configurable iterations, positions, depth, train/val split, resume support.

### Model Evaluation Tool — `eval_match` binary

Plays games between two NNUE models (or model vs. PST baseline) with
color swapping for fairness. Reports win/loss/draw and estimated Elo difference.

### Data Validation Utilities — `scripts/nnue/validate_sfen.py`

Validates SFEN training data: malformed line detection, score distribution
histogram, result balance (W/D/L), duplicate detection, multi-file summary.

### NNUE Quantization Export — `scripts/nnue/export_nnue_quantized.py`

Exports int16 (FT, scale=127) and int8 (hidden layers, scale=64) quantized
weights following the Stockfish NNUE convention. ~2x smaller than float32.

### Experiment Config Variants — `configs/nnue/small.yaml`, `configs/nnue/large.yaml`

Small (~640K params, ft_out_dim=64) for quick experiments.
Large (~20M params, ft_out_dim=512) for maximum strength.

### Training Visualization — TensorBoard in `trainer.py`

Logs train/val loss and learning rate per epoch. Enabled when `log_dir` is
set in config. Install with `pip install -e ".[viz]"`.

### Benchmarking Suite — `scripts/nnue/benchmark.py`

Measures inference speed (positions/sec), training throughput (samples/sec),
model parameter count, and GPU memory usage.
