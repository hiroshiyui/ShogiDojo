# TODO

## Open — Third paradigm: Bonanza-method supervised learning

A natural third paradigm to complement the NNUE and AlphaZero pipelines.
The Bonanza method (Hoki, 2006) trains evaluation weights by supervised
learning on professional game records (kifu), placing it at a distinct
point in the design space:

| Paradigm           | Data source        | Search at training? | Search at play? |
|--------------------|--------------------|---------------------|-----------------|
| Bonanza/supervised | Human games        | No                  | Yes (alpha-beta)|
| NNUE               | Engine self-play   | Yes (alpha-beta)    | Yes (alpha-beta)|
| AlphaZero          | Network self-play  | Yes (MCTS)          | Yes (MCTS)      |

**Why add it.** (a) Completes the spectrum from "learn from humans" →
"learn from engines" → "learn from yourself." (b) Low implementation
cost: reuse `encoding.py` + a simplified network (policy head only, no
value head, no MCTS, no replay buffer). Train with standard
cross-entropy on (position, move-played-by-pro) pairs. (c) Data is
freely available: professional Shogi kifu are archived online in large
quantities. (d) The resulting policy network could initialize the
AlphaZero self-play loop (warm-start instead of random weights), which
is known to accelerate early training.

Phased plan:

- **Phase 3.1 — Reference doc** *(done)*: `doc/bonanza/Bonanza_Reference.md`
  covers Bonanza's history (WCSC26 2006), the supervised classification
  formulation (policy cross-entropy + value MSE, modernized from the
  original logistic regression), CSA data format, three-paradigm
  comparison table, strengths and limitations, and references.
- **Phase 3.2 — Kifu parser + dataset** *(done)*: `src/shogi_dojo/bonanza/
  dataset.py` provides `parse_csa_string` / `parse_csa_file` (via
  python-shogi's CSA.Parser), `replay_game` (reconstructs the board
  state at every move and produces training samples), and
  `SupervisedDataset` (PyTorch map-style Dataset with `from_csa_files`
  / `from_games` constructors).
- **Phase 3.3 — Policy network + training** *(done)*: reuses
  `AlphaZeroNetwork` from the alphazero/ package. `BonanzaTrainer`
  runs epoch-based supervised training: masked log-softmax + NLL loss
  on the policy head against the pro's move index, MSE on the value
  head against the game outcome. Configurable `lambda_value` weight.
  Top-1 move-prediction accuracy logged per epoch.
  `configs/bonanza/{tiny,default}.yaml`, `train_bonanza.py` entry point.
  Tests under `tests/bonanza/test_bonanza.py` (20 tests) cover CSA
  parsing, game replay (sample shapes, min/max ply, value targets,
  legal masks), dataset construction, and an end-to-end smoke test
  (loop runs, checkpoint saved, parameters change, both YAML configs
  parse).
- **Phase 3.4 — Move-ordering integration** *(deferred — significant
  Rust infrastructure)*: use the trained policy network to order moves
  in the existing gensfen alpha-beta search. Would require implementing
  ResNet forward pass in Rust (convolutions, BatchNorm, residual
  blocks) plus a serialization format for the two-headed model.
  Estimated scope: ~500–1,000 lines of Rust, comparable to the
  existing NNUE inference code. Pedagogically valuable but infra-heavy;
  skipped for the Python-first survey. If pursued later, the simplest
  approach is a Python demo (small alpha-beta implementation + neural
  move ordering) to measure nodes-to-depth before committing to a
  Rust port.
- **Phase 3.5 — AlphaZero warm-start** *(done)*:
  `AlphaZeroTrainer` now accepts a `model.warm_start` config key
  pointing at any AlphaZeroNetwork-compatible checkpoint (Bonanza or
  AlphaZero). Because both paradigms use the same architecture, the
  state_dict loads directly without conversion. Tests under
  `tests/alphazero/test_trainer.py::TestAlphaZeroTrainerWarmStart`
  (2 tests) verify the checkpoint's weights are copied exactly and
  that omitting the key leaves random initialization.
- **Phase 3.6 — Three-way comparison** *(infrastructure done;
  execution deferred)*: `src/shogi_dojo/alphazero/evaluation.py`
  provides `MatchResult` (wins/losses/draws with Elo-diff
  calculation) and `run_match` (plays N games between two
  AlphaZero-family networks, color-swapped, no Dirichlet noise).
  CLI at `scripts/alphazero/match.py` takes two checkpoint paths and
  reports win rate + Elo estimate. Tests under
  `tests/alphazero/test_evaluation.py` (14 tests) cover MatchResult
  arithmetic (win-rate and Elo formulas at boundary values) and
  integration (game count rounding, color balancing, short-limit
  draws). Actual three-way benchmark run (NNUE vs AlphaZero vs
  Bonanza on equal compute) is a research exercise left to the team
  — it requires GPU-days of training per paradigm. Note: the existing
  Rust `eval_match` tool handles NNUE-vs-NNUE matches; for
  cross-paradigm matches (NNUE vs AlphaZero), a new Python harness
  would be needed to drive both simultaneously.

### Other paradigms worth knowing about

These are not planned for implementation but are noted here for
literature awareness:

- **dlshogi** — AlphaZero-family but with distinct Shogi-specific
  engineering: virtual loss for GPU-batched MCTS, USI protocol
  integration, tournament-grade time management. Won WCSC multiple
  times. The benchmark to study if the AlphaZero implementation
  ever targets competitive strength. Code:
  `github.com/TadaoYamaoka/DeepLearningShogi`.
- **Transformer-based evaluation** — emerging research direction
  replacing the CNN trunk with Vision Transformer / attention over
  board squares and hand pieces as tokens. No major tournament
  wins yet, but Shogi's drop rule (pieces appear/disappear between
  hand and board) is a natural fit for attention. A tiny Transformer
  policy/value network compared against the ResNet on the same
  self-play data would be a publishable contribution.
- **Pure MCTS (no neural network)** — UCT with random rollouts
  instead of neural evaluation. Historically important (competitive
  in Go before AlphaGo), now superseded. The existing `mcts.py` can
  demonstrate this directly: pass an `evaluate_fn` that returns
  `(uniform_prior, 0.0)` and the search degrades gracefully to
  UCT. Useful as a baseline.

---

## Open — Research directions

Now that ShogiDojo has working implementations of both NNUE and
AlphaZero, several research directions are available. They are
ordered roughly by expected impact-per-effort for an academic team
with limited GPU resources.

### 1. Policy distillation: AlphaZero → NNUE

**Idea.** Train an AlphaZero-style network via self-play to produce
high-quality move-selection data (the MCTS visit distributions),
then use those distributions as training targets for the NNUE
pipeline's gensfen tool. The NNUE model cannot predict a *policy*
directly, but it can be trained on positions whose *scores* were
generated by an MCTS-informed evaluator — effectively distilling
AlphaZero's search knowledge into a fast NNUE evaluation.

**Why interesting.** Combines AlphaZero's data-quality advantage
(no need for an external strong engine) with NNUE's deployment
advantage (sub-millisecond inference on CPU). This is an active
research direction in the broader community.

**Concrete steps.** (a) Run AlphaZero self-play at moderate depth
(100+ sims) to produce a set of (SFEN, MCTS-value) pairs. (b) Feed
those into the NNUE training pipeline as if they were gensfen output
(the value replaces the alpha-beta score). (c) Compare the resulting
NNUE against one trained on gensfen-only data via eval_match.

### 2. Evaluation-match promotion gating for AlphaZero

**Idea.** The current `AlphaZeroTrainer` saves every iteration's
weights and uses them unconditionally for the next iteration's
self-play. Standard practice (LCZero, KataGo) is to *gate* promotion:
play an evaluation match between the new and current-best networks,
and only promote if the new network wins by a margin (≥55% win rate
in 40+ games).

**Why interesting.** Without gating, a bad training step can produce
a weaker network whose self-play data then drags the next iteration
down — a downward spiral. Gating prevents this. It is described in
`AlphaZero_Training_Reference.md` §6 but not yet implemented.

**Concrete steps.** (a) Add a `run_match(network_a, network_b,
num_games, num_simulations)` function to `rules.py` or a new
`evaluation.py`. (b) Integrate into the trainer's iteration loop
after each training phase. (c) Track best-network separately from
latest-network.

### 3. Auxiliary training targets (KataGo-style)

**Idea.** Add auxiliary heads to the AlphaZero network that predict
additional targets beyond policy and value:

- **Opponent's reply policy** — what move will the opponent play?
- **Ownership prediction** — which squares will be controlled by
  which side at the end of the game? (adapted from Go to Shogi)
- **Score prediction** — predict the *margin* of victory in
  centipawns, not just win/loss/draw.

**Why interesting.** KataGo showed that auxiliary targets provide
additional gradient signal that accelerates learning, especially
early in training when the value and policy targets are noisy.
The score-prediction head is particularly interesting because it
bridges the AlphaZero and NNUE paradigms — NNUE trains on scores
directly, while standard AlphaZero only trains on outcomes.

**Concrete steps.** (a) Add auxiliary head(s) to `network.py`.
(b) Extend `self_play.py` to record the additional targets.
(c) Extend `alphazero_loss` with the new terms. (d) Compare
training efficiency (iterations-to-strength) with and without
auxiliary heads.

### 4. Hybrid search: NNUE evaluation inside MCTS

**Idea.** Replace the AlphaZero network's value head with the
trained NNUE evaluator for leaf evaluation, while keeping the
policy head for MCTS guidance. This is a simplified version of
what dlshogi and similar hybrid engines do.

**Why interesting.** The NNUE evaluator is ~100,000× faster than
a full AlphaZero forward pass. If its positional judgment is
comparable, the MCTS can run orders of magnitude more simulations
in the same wall time, potentially producing stronger play even
with a weaker (but faster) evaluation.

**Concrete steps.** (a) Load a trained NNUE checkpoint and wrap
it as a value-only evaluator. (b) Modify `make_evaluate_fn` in
`rules.py` to use the AlphaZero policy head but the NNUE value.
(c) Run eval_match comparisons: pure-AlphaZero vs hybrid vs
pure-NNUE-alpha-beta.

### 5. Opening book from self-play statistics

**Idea.** Aggregate the MCTS visit distributions from many
self-play games at early plies to build a probabilistic opening
book. Positions with consistent high-visit moves across many games
indicate strong opening preferences.

**Why interesting.** A data-driven opening book requires no
human curation. The team can visualize which openings the network
"likes" and compare them against known joseki (standard Shogi
openings). This is more accessible than playing strength as a
research output — it produces human-readable artifacts.

**Concrete steps.** (a) During self-play, dump (SFEN, visit_counts)
for the first N plies to a separate file. (b) Aggregate across
games: for each SFEN, average the visit distributions. (c) Build
a simple tree viewer that shows the top 3 moves at each position
with their visit percentages.

### 6. Quantitative paradigm comparison

**Idea.** Run both pipelines on identical compute budgets and
measure playing strength via eval_match at each checkpoint.
The comparison in `AlphaZero_vs_NNUE.md` is qualitative; a
quantitative version with actual Elo curves would be the
flagship output of this research survey.

**Why interesting.** The literature claims that NNUE reaches
usable strength much faster per GPU-hour while AlphaZero
eventually surpasses it given enough compute. Verifying (or
contradicting) this claim on Shogi with our own code is a
genuine research contribution.

**Concrete steps.** (a) Define a fixed compute budget (e.g.,
1 GPU-day). (b) Train NNUE (gensfen + train iterations) and
AlphaZero (self-play iterations) within that budget. (c) At
regular intervals, freeze a checkpoint and run 200-game
eval_match against a fixed baseline (PST evaluator). (d) Plot
Elo vs wall-clock-time for both paradigms.

### 7. Board-size reduction for faster experimentation

**Idea.** Implement a 5×5 or 7×7 mini-Shogi variant (e.g.,
Minishogi / Gogo Shogi) where games are shorter and the state
space is smaller. Both NNUE and AlphaZero pipelines would need
minor adaptation (feature dimensions, move encoding), but the
core algorithms are unchanged.

**Why interesting.** Smaller boards make experiments 10–100×
faster, enabling the team to iterate on algorithmic ideas
(auxiliary heads, hybrid search, distillation) before committing
to full 9×9 Shogi runs. Many research groups use reduced-board
variants for exactly this purpose.

**Concrete steps.** (a) Define a `MiniShogi` rule set (5×5 board,
subset of pieces). (b) Adapt `encoding.py` and `constants.py` for
the smaller board. (c) Verify both training pipelines run on the
reduced variant. (d) Use it as a fast testbed for items 1–6 above.

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

### AlphaZero Survey — `src/shogi_dojo/alphazero/`, `doc/alphazero/`

Educational research survey of the AlphaZero family of game-playing
systems, built under `alphazero/` subdirectories in parallel with NNUE.
Docs lead, code follows. Delivered in nine phases:

1. **Skeleton** — `alphazero/` subdirs and Python sub-package.
2. **Foundational doc** — `doc/alphazero/AlphaZero_Reference.md`:
   history, architecture, Shogi encoding, MCTS+PUCT, self-play loop,
   loss function, NNUE comparison, glossary.
3. **Training reference** — `doc/alphazero/AlphaZero_Training_Reference.md`:
   self-play methodology, replay-buffer dynamics, loss decomposition,
   hyperparameters, training loop, evaluation gating, resource estimates,
   failure modes, NNUE comparison.
4. **Encoding** — `encoding.py` + 69 tests: 45-channel input tensor,
   9×9×139 action-space encode/decode, legal-move mask, SFEN bridge.
5. **Network** — `network.py` + 19 tests: `AlphaZeroNetwork` (ResNet +
   policy/value heads, ~336K params), `masked_log_softmax`,
   `alphazero_loss`.
6. **MCTS** — `mcts.py` + 30 tests: game-agnostic PUCT MCTS with
   Dirichlet noise, temperature-based move selection.
7. **Self-play + trainer**:
   - `rules.py` + 30 tests: python-shogi adapter for MCTS callables.
   - `self_play.py` + 23 tests: `play_game`, `ReplayBuffer`.
   - `trainer.py` + 6 tests: `AlphaZeroTrainer` iteration loop.
   - `train_alphazero.py` entry point,
     `configs/alphazero/{tiny,default}.yaml`.
8. **Comparison and reading list** —
   `doc/alphazero/AlphaZero_vs_NNUE.md` (operational comparison),
   `doc/alphazero/AlphaZero_Reading_List.md` (annotated bibliography).
9. **README integration** — `README.md` AlphaZero Survey section,
   `CLAUDE.md` architecture subsections for both paradigms.

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
