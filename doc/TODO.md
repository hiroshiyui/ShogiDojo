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
- **Phase 2.4 — Encoding code + tests** *(done)*:
  `src/shogi_dojo/alphazero/encoding.py` provides `Position` and
  `Move` dataclasses, `encode_position` (45-channel input tensor
  from the side-to-move's perspective), `encode_move` /
  `decode_move` over the paper's 9×9×139 = 11,259 action space,
  `legal_move_mask` for masking illegal logits before softmax, and
  a `position_from_sfen` convenience wrapper. Tests under
  `tests/alphazero/test_encoding.py` cover constants, shape,
  scalar planes, per-piece/hand planes, perspective flip,
  round-tripping for queen/knight/drop/promoting moves under both
  sides, error handling, legal-move mask shape/contents, and
  cross-color plane consistency (69 tests total). Shogi facts
  (piece type IDs, board dimensions) are imported from
  `shogi_dojo.nnue.constants`; the SFEN parser is shared via a
  local import to avoid duplicating 200 lines.
- **Phase 2.5 — Network code + tests** *(done)*:
  `src/shogi_dojo/alphazero/network.py` implements `ResidualBlock`
  and `AlphaZeroNetwork`: stem 3×3 conv (45 → F filters) → N
  residual blocks (each BN + ReLU + skip) → policy head (1×1 conv
  to 139 planes, raw logits) + value head (1×1 conv → flatten → FC →
  tanh scalar). Defaults (`num_blocks=4`, `num_filters=64`,
  `value_head_hidden=64`) give ~336K parameters — tiny by AlphaZero
  standards but trainable on a single CPU. Helpers
  `masked_log_softmax` (apply legal-move mask before softmax with
  ``-1e9`` fill) and `alphazero_loss` (value-MSE + policy
  cross-entropy with configurable term weights) round out the module.
  Kaiming-normal init mirrors the NNUE code style. Tests under
  `tests/alphazero/test_network.py` (19 tests) cover residual-block
  shape preservation, identity-path representability, default
  construction, parameter-count sanity, forward-pass shape/dtype/
  value-range/finiteness under several batch sizes, train vs eval
  BatchNorm behavior, gradient flow to every parameter, and the
  mask/loss helpers.
- **Phase 2.6 — MCTS code + tests** *(done)*:
  `src/shogi_dojo/alphazero/mcts.py` implements readable PUCT MCTS:
  `Edge` (prior/N/W/Q/child), `Node` (state + dict of edges +
  terminal flag), `puct_score`, `select_child`, `apply_dirichlet_noise`,
  `MCTS.search(root_state, num_simulations, dirichlet_alpha,
  dirichlet_epsilon, rng)` with the standard four-phase simulation
  (select → expand → evaluate → backup with sign flip), plus helpers
  `visit_counts`, `policy_target(root, temperature)`, and
  `sample_move(root, temperature, greedy)`. The search is
  game-agnostic: the caller passes `evaluate_fn`, `legal_moves_fn`,
  `apply_fn`, and `is_terminal_fn` callables. Default `c_puct=1.5`,
  no subtree carryover between searches. Tests under
  `tests/alphazero/test_mcts.py` (30 tests) use a hand-built 2-ply
  zero-sum toy game with hardcoded terminal values to verify PUCT
  math, select_child behavior, Dirichlet noise (priors still sum to
  1, ε=0 is a no-op, ε=1 replaces priors), search basics (terminal
  root, root expansion, visit-sum = num_simulations, visits
  concentrate on higher-Q move), the perspective sign-flip on backup,
  Dirichlet noise spreading visits averaged over 30 seeds in a
  flat-Q variant game, and the helpers (visit_counts, policy_target
  at multiple temperatures, sample_move greedy/stochastic).
- **Phase 2.7 — Self-play + trainer + smoke test** *(in progress)*:

  - **2.7a — python-shogi adapter** *(done)*:
    `src/shogi_dojo/alphazero/rules.py` wires python-shogi to the
    game-agnostic MCTS interface. `python-shogi` added to
    `pyproject.toml`. Provides piece-type translation (python-shogi
    uses 1-indexed IDs starting at PAWN=1; we use 0-indexed from
    PAWN=0), `board_to_position` / `shogi_move_to_move` /
    `move_to_shogi_move`, the four MCTS callables (`legal_moves_fn`,
    `apply_fn` via SFEN round-trip since python-shogi lacks `.copy()`,
    `is_terminal_fn` with a 512-ply default limit and sennichite →
    draw), and `make_evaluate_fn` that bridges an `AlphaZeroNetwork`
    into an MCTS-compatible evaluator (forward pass under
    `torch.no_grad()`, legal-move masking, log-softmax, keyed by
    original `shogi.Move`). Tests under `tests/alphazero/test_rules.py`
    (30 tests) cover piece-type bijection, starting-position
    translation, move round-trip (ordinary / promoting / drop),
    ``legal_moves_fn``, ``apply_fn`` purity, terminal detection
    (including a verified two-gold corner checkmate), and the
    evaluator's plumbing (keys, sum-to-one, value range, training-
    mode restoration, no grad leaks).
  - **2.7b — self-play + replay buffer** *(done)*:
    `src/shogi_dojo/alphazero/self_play.py` provides
    `TrainingExample` (`input_tensor`, `policy_target`,
    `value_target`, `legal_mask`), `play_game` (one full self-play
    game with MCTS at every move, Dirichlet noise at root, τ=1 for
    the first 30 plies then argmax, terminal z backfill per STM
    perspective), and `ReplayBuffer` (FIFO `deque(maxlen=capacity)`,
    uniform random sampling with replacement, returns a dict of
    stacked tensors ready for training). Tests under
    `tests/alphazero/test_self_play.py` (23 tests) cover
    `_backfill_z` for all terminal cases, `_policy_dict_to_tensor`
    placement and sum, replay-buffer construction validation,
    `add`/`add_many` length invariants, FIFO eviction, `is_full`
    flip, `sample` shapes/dtypes/empty-buffer error/batch-larger-
    than-buffer, and `play_game` integration (returns
    `TrainingExample`s with correct shapes, short move_limit yields
    all-zero z values, terminal starting board returns empty list,
    policy targets sum to ~1, legal masks non-empty at every step).
  - **2.7c — trainer + configs + scripts + smoke test** *(done)*:
    `src/shogi_dojo/alphazero/trainer.py` (`AlphaZeroTrainer`:
    config-driven iteration loop — self-play → buffer → Adam
    gradient steps → checkpoint), `configs/alphazero/tiny.yaml` (1
    game × 20 sims × 10 iterations — runs in minutes on CPU) and
    `configs/alphazero/default.yaml` (20 games × 100 sims × 50
    iterations — hours-scale), top-level `train_alphazero.py` entry
    point. Evaluation-match promotion gating deferred to a later
    phase. Tests under `tests/alphazero/test_trainer.py` (6 tests):
    construction from config dict, end-to-end smoke test (2
    iterations, 1 game each, move_limit=6 — completes in seconds,
    buffer grows, checkpoint saved with model+optimizer state,
    network parameters change after training), and both YAML configs
    parse and construct correctly.
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
