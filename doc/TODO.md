# TODO

## Open — Shogi rule coverage gaps

The 2026-04-14 rule-compliance audit surfaced three gaps that remain.
None blocks training, but fixing them would improve data fidelity:

- [ ] **Try rule (入玉宣言勝ち, "entering king")** — not detected. A king that
  reaches the opponent's promotion zone under the usual point-count
  conditions should be declared a winner; `game.rs` currently lets such
  games drift to the `MAX_PLY` draw.
- [ ] **Impasse / 持将棋 (jishōgi)** — not detected. When both kings enter
  opposing promotion zones and neither side can mate, the game is
  adjudicated by point count, not left to time out at ply 400.
- [ ] **Defense-in-depth uchi-fu-zume** — currently delegated to shogi-rs
  via `MoveError::Uchifuzume`. `game::tests::uchi_fu_zume_pawn_drop_is_filtered`
  locks in the behaviour end-to-end, but we do not have a local,
  shogi-rs-independent implementation.

---

## Open — Bonanza-method paradigm: remaining work

The Bonanza-method paradigm (supervised policy + value training from
professional kifu) landed phases 3.1, 3.2, 3.3, 3.5, and 3.6
infrastructure, plus the v2 full-archive run (2026-10-23) — see the
Completed section for details. Open items:

- [ ] **Phase 3.4 — Move-ordering integration with gensfen alpha-beta**
  *(deferred — significant Rust infrastructure)*. Would require
  implementing a ResNet forward pass in Rust (convolutions, BatchNorm,
  residual blocks) plus a serialization format for the two-headed
  model. Estimated scope: ~500–1,000 lines of Rust, comparable to the
  existing NNUE inference code. Pedagogically valuable but infra-heavy;
  skipped for the Python-first survey. If pursued later, the simplest
  approach is a Python demo (small alpha-beta implementation + neural
  move ordering) to measure nodes-to-depth before committing to a
  Rust port.
- [ ] **Phase 3.6 — Three-way quantitative benchmark run**. The match
  infrastructure (`src/shogi_dojo/alphazero/evaluation.py` +
  `scripts/alphazero/match.py`) is complete. The actual benchmark —
  train NNUE, AlphaZero, and Bonanza on equal compute budgets and
  plot Elo curves — is a research exercise requiring GPU-days per
  paradigm. Note: cross-paradigm matches (NNUE vs AlphaZero) would
  need a new Python harness to drive both simultaneously; the
  existing Rust `eval_match` handles only NNUE-vs-NNUE.
- [ ] **Finite Elo estimate for v2.** v2 beat v1 208–0, so v1 is too
  weak a reference to measure v2 (only a lower bound of ~+735 Elo).
  Re-run against a closer opponent (a future larger net, or a v2
  checkpoint from epoch 1) and/or at `--temperature 0.5` so games
  differ, and report Elo with a confidence interval.
- [ ] **Larger network (15×192, ~10M params) on the packed archive.**
  v2's validation accuracy still rose ~0.7 points per epoch, so it is
  not saturated. Benchmarked at ~2,900 samples/s with AMP on an RTX
  3060, so 4 epochs take ~3 days; no preprocessing needed (reuse
  `data/packed/wdoor_all`).
- [ ] **Filter training games by player rating.** Floodgate CSA files
  carry player ratings as comment lines (`'black_rate:<name>:3833.0`,
  `'white_rate:...`); training only on games where both players
  exceed a threshold should raise move quality. Needs the rating
  recorded at preprocessing time (e.g. a per-game field in the shard
  `meta`) so the filter can be changed without re-preprocessing.
- [ ] **Investigate data augmentation.** A left-right mirror doubles
  the data, but Shogi's starting position is not mirror-symmetric
  (rook and bishop swap sides), so mirrored games are legal but come
  from a different opening. Measure whether it helps before adopting.
- [ ] **AlphaZero self-play warm-started from v2.** Use
  `checkpoints/bonanza_v2/bonanza_epoch_0004.pt` as `model.warm_start`
  with a 10×128 AlphaZero config; pair with promotion gating (research
  direction 2) so self-play cannot drift below the v2 starting point.

### Other paradigms worth knowing about

Literature awareness, not planned for implementation:

- [ ] **dlshogi** — AlphaZero-family with Shogi-specific engineering
  (virtual loss, USI protocol, WCSC-grade time management). The
  benchmark to study if the AlphaZero implementation ever targets
  competitive strength. Code: `github.com/TadaoYamaoka/DeepLearningShogi`.
- [ ] **Transformer-based evaluation** — replacing the CNN trunk with
  Vision Transformer / attention over board squares and hand pieces
  as tokens. No major tournament wins yet, but Shogi's drop rule fits
  attention naturally. A tiny-Transformer vs ResNet comparison on the
  same self-play data would be a publishable contribution.
- [ ] **Pure MCTS (no neural network)** — UCT with random rollouts,
  superseded by neural-guided MCTS. The existing `mcts.py` can
  demonstrate this: pass an `evaluate_fn` returning
  `(uniform_prior, 0.0)` and search degrades to UCT. Useful as a
  baseline.

---

## Open — Research directions

Seven research directions available once the three paradigms are
running, ordered roughly by expected impact-per-effort for an
academic team with limited GPU resources.

- [ ] **1. Policy distillation: AlphaZero → NNUE.** Train an AlphaZero
  network via self-play to produce high-quality (SFEN, MCTS-value)
  pairs, then feed them into the NNUE training pipeline as if they
  were gensfen output. Compare the distilled NNUE against a gensfen-
  only NNUE via `eval_match`. Combines AlphaZero's data-quality
  advantage with NNUE's deployment advantage.

- [ ] **2. Evaluation-match promotion gating for AlphaZero.** Integrate
  `run_match` (already in `evaluation.py`) into the training loop:
  play 40 games after each iteration, promote only if win rate ≥ 55%.
  Documented in `AlphaZero_Training_Reference.md` §6 but not wired in.
  Prevents downward spirals from bad training steps.

- [ ] **3. Auxiliary training targets (KataGo-style).** Add auxiliary
  heads predicting opponent's reply policy, ownership, or score margin.
  KataGo showed these accelerate early training. The score-prediction
  head is particularly interesting because it bridges AlphaZero and
  NNUE — NNUE trains on scores directly. Steps: add auxiliary head(s)
  to `network.py`, extend `self_play.py` to record the targets, extend
  `alphazero_loss` with the new terms, compare training efficiency
  with/without.

- [ ] **4. Hybrid search: NNUE evaluation inside MCTS.** Replace the
  AlphaZero value head with the NNUE evaluator at leaves, keep the
  policy head for MCTS guidance. NNUE is ~100,000× faster than the
  full ResNet forward pass, so MCTS can run orders of magnitude more
  simulations in the same wall time. Steps: load a trained NNUE as a
  value-only evaluator, modify `make_evaluate_fn` in `rules.py`, run
  `eval_match` comparisons.

- [ ] **5. Opening book from self-play statistics.** Aggregate MCTS
  visit distributions at early plies across many self-play games to
  build a probabilistic opening book. Positions with consistent high-
  visit moves indicate strong opening preferences. Produces human-
  readable artifacts — more accessible as a research output than
  playing strength. Steps: dump (SFEN, visit_counts) at early plies
  during self-play, aggregate, build a tree viewer.

- [ ] **6. Quantitative three-paradigm comparison.** Fixed compute
  budget (e.g. 1 GPU-day), train NNUE / AlphaZero / Bonanza within
  that budget, freeze checkpoints at regular intervals, run 200-game
  `eval_match` against a fixed baseline (PST), plot Elo vs wall-clock
  for each paradigm. Verifies or contradicts the literature claim
  that NNUE reaches usable strength much faster per GPU-hour while
  AlphaZero eventually surpasses it. The flagship quantitative output
  of this survey.

- [ ] **7. Board-size reduction for faster experimentation.** Implement
  5×5 Minishogi or 7×7 variants. Games are shorter, state space is
  smaller, experiments are 10–100× faster. Useful testbed for items
  1–6 above. Steps: define a `MiniShogi` rule set, adapt `encoding.py`
  and `constants.py`, verify both training pipelines run, use as the
  iteration platform for algorithmic experiments before committing
  to full 9×9 runs.

---

## Completed

### Bonanza-method paradigm — Done (2026-04-16)

Supervised policy + value training from professional kifu records,
filling the "learn from humans" slot in the three-paradigm survey.
Covers Phases 3.1–3.3 (reference doc, CSA parser, dataset, supervised
trainer), 3.5 (AlphaZero warm-start from Bonanza checkpoints — state_dicts
are directly compatible because both paradigms reuse `AlphaZeroNetwork`),
and 3.6 infrastructure (`evaluation.py` match tool, `match.py` CLI).
Also includes a streaming `IterableDataset` that bounds memory for the
full ~167K-file shogi-server wdoor archive. Entry point
`train_bonanza.py`, configs at `configs/bonanza/{tiny,default}.yaml`,
reference at `doc/bonanza/Bonanza_Reference.md`, 29 tests in
`tests/bonanza/`. Remaining work tracked in the "Bonanza-method
paradigm: remaining work" section above.

### Bonanza v2: full-archive training — Done (2026-10-23)

Trained a 10×128 network (3.04M params) on all 16 shogi-server wdoor
archives (2011–2026: 1,571,095 games, 188.4M positions) in 24 h 17 min
on one RTX 3060, reaching 54.33% top-1 move accuracy on held-out games
(v1: 44.46% on training data) and beating v1 208–0. Enabled by packed
preprocessing (`packed.py` + `scripts/bonanza/preprocess_kifu.py`:
compact memory-mapped shards, ~90K positions/s loading instead of
~2.2K, crash-tolerant multi-process build) and trainer options in
`configs/bonanza/v2.yaml` (mixed precision, AdamW, warmup + cosine LR,
game-level validation split with best-model tracking). `match.py` now
detects each checkpoint's architecture, so differently-sized networks
can play each other. Artifacts committed as `bonanza_v2.onnx` and
`checkpoints/bonanza_v2/`; results in `doc/bonanza/Bonanza_Reference.md`
"v2 results".

### AlphaZero Survey — Done (2026-04-16)

Educational research survey of the AlphaZero family, built under
`alphazero/` subdirectories in parallel with NNUE. Delivered in nine
phases (skeleton, foundational doc, training reference, encoding with
69 tests, network with 19 tests, MCTS with 30 tests, python-shogi
adapter / self-play / trainer with 30+23+6 tests, NNUE comparison doc
and reading list, README integration). End-to-end runnable via
`python train_alphazero.py`. Entry point at repo root, code at
`src/shogi_dojo/alphazero/`, docs at `doc/alphazero/`, configs at
`configs/alphazero/{tiny,default}.yaml`, tests at `tests/alphazero/`.

### Rust gensfen Tool — Done

Standalone Rust tool for NNUE training data generation via self-play.
Alpha-beta search with iterative deepening, PST and NNUE evaluation,
opening randomization, multi-threaded generation (rayon), text SFEN
output. Lives at `tools/nnue/gensfen/`.

### CI/CD Pipeline — Done

GitHub Actions on every push/PR to `main`: Python (ruff + pytest) and
Rust (fmt + clippy + test). Workflow at `.github/workflows/ci.yml`.

### NNUE Iterative Training Orchestration — Done

`scripts/nnue/train_loop.py` automates the self-improvement loop:
gensfen → train → export → regenerate with NNUE eval. Configurable
iterations, positions, depth, train/val split, resume support.

### Model Evaluation Tool — Done

`eval_match` binary plays games between two NNUE models (or model vs
PST baseline) with color swapping for fairness. Reports win/loss/draw
and estimated Elo difference. Lives alongside gensfen.

### Data Validation Utilities — Done

`scripts/nnue/validate_sfen.py` validates SFEN training data:
malformed line detection, score distribution histogram, result balance
(W/D/L), duplicate detection, multi-file summary.

### NNUE Quantization Export — Done

`scripts/nnue/export_nnue_quantized.py` exports int16 (Feature
Transformer, scale=127) and int8 (hidden layers, scale=64) quantized
weights following the Stockfish NNUE convention. ~2× smaller than
float32.

### Experiment Config Variants — Done

`configs/nnue/small.yaml` (~640K params, `ft_out_dim=64`) for quick
experiments, `configs/nnue/large.yaml` (~20M params, `ft_out_dim=512`)
for maximum strength, alongside the default `configs/nnue/default.yaml`.

### Training Visualization — Done

TensorBoard logging in `src/shogi_dojo/nnue/trainer.py`: train/val
loss and learning rate per epoch. Enabled when `log_dir` is set in
config. Install with `pip install -e ".[viz]"`.

### Benchmarking Suite — Done

`scripts/nnue/benchmark.py` measures inference speed (positions/sec),
training throughput (samples/sec), model parameter count, and GPU
memory usage.
