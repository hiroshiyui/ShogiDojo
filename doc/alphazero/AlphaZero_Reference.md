# AlphaZero Reference Guide

A comprehensive technical reference for the AlphaZero family of game-playing
systems, written from the perspective of researchers familiar with the
ShogiDojo NNUE pipeline. This document is aimed at team members who have
read [`doc/nnue/NNUE_Reference.md`](../nnue/NNUE_Reference.md) and want to
understand the alternative paradigm — what AlphaZero is, why it matters for
computer Shogi, and how the moving parts fit together before we attempt any
implementation.

> **Status — research survey in progress.** This document is part of an
> ongoing literature survey. It is intentionally pedagogical and does not
> describe a working implementation in this repository. Code under
> `src/shogi_dojo/alphazero/` will be added in later phases and will follow
> the conventions established here.

## Table of Contents

1. [Why Survey AlphaZero?](#1-why-survey-alphazero)
2. [Historical Context](#2-historical-context)
3. [System Overview](#3-system-overview)
4. [Network Architecture](#4-network-architecture)
5. [Input Encoding for Shogi](#5-input-encoding-for-shogi)
6. [Move (Action) Encoding for Shogi](#6-move-action-encoding-for-shogi)
7. [Monte Carlo Tree Search with PUCT](#7-monte-carlo-tree-search-with-puct)
8. [Self-Play and the Training Loop](#8-self-play-and-the-training-loop)
9. [Loss Function](#9-loss-function)
10. [Comparison with NNUE](#10-comparison-with-nnue)
11. [Glossary](#11-glossary)
12. [References](#12-references)

---

## 1. Why Survey AlphaZero?

ShogiDojo's primary effort is HalfKAv2 NNUE training: a small (~5M parameter)
network that runs millions of evaluations per second inside a classical
alpha-beta search. NNUE is the technology that powers the strongest
publicly available chess and Shogi engines today (Stockfish, dlshogi-like
hybrids, etc.).

AlphaZero takes a very different design path:

- **One large network** (~tens of millions of parameters) replaces both the
  hand-crafted search heuristics and the static evaluation.
- **Monte Carlo Tree Search (MCTS)** guided by the network's policy output
  replaces alpha-beta search.
- **Self-play from scratch**, with no human games or hand-crafted features,
  produces all the training data.

Surveying AlphaZero helps the team in three ways:

1. **Vocabulary**: Modern computer-game-playing literature is bilingual —
   it speaks both NNUE/alpha-beta and AlphaZero/MCTS. Reading recent
   papers (KataGo, Leela Chess Zero, dlshogi) requires understanding both
   sides.
2. **Design contrast**: Many NNUE design choices (sparse features, king
   buckets, ClippedReLU, quantization) make sense *because* the inference
   budget is tiny. Studying a system designed under the opposite
   assumption clarifies why our choices matter.
3. **Future research**: Hybrid systems (NNUE-evaluated MCTS, distillation
   from AlphaZero-trained policies into NNUE-style nets) are an active
   research area. A working understanding of both paradigms is a
   prerequisite to pursuing them.

This survey does **not** aim to produce a competitive Shogi engine. It aims
to produce a clear, beginner-friendly reference and a small didactic
implementation that the team can read, run, and reason about.

---

## 2. Historical Context

The AlphaZero name refers to a specific 2017 system from DeepMind, but the
ideas evolved across three papers:

| Year | System              | Key Contribution                                                |
|------|---------------------|------------------------------------------------------------------|
| 2016 | **AlphaGo**         | First neural-network MCTS to defeat a top human Go player. Used a *policy* network (trained on human games) and a *value* network (trained on self-play). Required hand-engineered features and a separate fast rollout policy. |
| 2017 | **AlphaGo Zero**    | Removed human data and hand-engineered features. Used a *single* network with two heads (policy + value), trained purely from self-play, with MCTS used both for action selection and as the training target for the policy. |
| 2017 | **AlphaZero**       | Generalized AlphaGo Zero to chess, Shogi, and Go simultaneously, with the same architecture and minimal game-specific code. |
| 2019 | **MuZero** (related)| Removed the requirement that the model knows the rules of the game; learned a dynamics model alongside policy and value. Out of scope for this survey but worth knowing exists. |

The 2017 AlphaZero paper explicitly demonstrated that the system could
learn Shogi from scratch in roughly nine hours of TPU-cluster time and
defeat the (then-)strongest Shogi engine, Elmo. This is the headline
result that motivates Shogi researchers to study AlphaZero specifically.

> **Vocabulary note.** "AlphaZero" is sometimes used loosely in the
> literature to mean "any AlphaGo-Zero–style policy/value MCTS system."
> Open-source projects in this family (Leela Zero, Leela Chess Zero,
> KataGo, dlshogi, AobaZero) follow the same recipe with various
> engineering improvements. We will use **AlphaZero-style** when we mean
> the broader family.

---

## 3. System Overview

An AlphaZero-style system is built around three components that interact
in a continuous loop:

```
                      ┌──────────────────────┐
                      │      Replay Buffer    │
                      │  (state, π, z) tuples │
                      └─────────┬────────────┘
                                │ sample mini-batches
                                ▼
                      ┌──────────────────────┐
                      │      Trainer          │
                      │  (gradient descent on │
                      │   policy + value loss)│
                      └─────────┬────────────┘
                                │ updated weights
                                ▼
                      ┌──────────────────────┐
                      │      Network          │
                      │  (state → π_net, v)   │
                      └─────────┬────────────┘
                                │ used inside MCTS
                                ▼
        ┌─────────────────────────────────────────────────┐
        │                  Self-Play Worker                │
        │                                                  │
        │   For each game:                                 │
        │     1. Run MCTS at every position                │
        │     2. Pick a move from MCTS visit counts        │
        │     3. Record (state, MCTS visit distribution π) │
        │   At game end:                                   │
        │     4. Backfill terminal result z into all       │
        │        positions of that game                    │
        │     5. Push (state, π, z) tuples to replay buffer│
        └─────────────────────────────────────────────────┘
                                │
                                └──→ feeds replay buffer
```

Three things to notice:

- The network is **never trained on raw game outcomes alone**. It learns
  to imitate the *MCTS visit distribution* π for the policy head, and the
  *terminal result* z for the value head. MCTS plays the role of a
  "policy improvement operator" — running search from any position
  produces a policy that is provably better than the network's raw output.
- Training is **off-policy**: the network being updated may be slightly
  newer than the network that generated the data. This is acceptable in
  practice and standard in modern reimplementations.
- There is **no dataset** in the traditional sense. Data is continuously
  produced by self-play with the latest network, and old data is aged out
  of the replay buffer. This is why AlphaZero is so compute-intensive:
  the network and the data co-evolve.

---

## 4. Network Architecture

### High-Level Shape

A single convolutional residual network (ResNet) reads the board state and
produces two outputs (the "two-headed" architecture):

```
            Board state (planes of 9×9 features)
                          │
                          ▼
              ┌──────────────────────┐
              │   Initial conv block │   3×3 conv → BatchNorm → ReLU
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Residual tower      │   N residual blocks
              │   (N typically 6–40)  │   each: conv → BN → ReLU → conv → BN → add → ReLU
              └──────────┬───────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   ┌─────────────────┐       ┌─────────────────┐
   │  Policy Head    │       │   Value Head    │
   │  conv 1×1 → FC  │       │  conv 1×1 → FC  │
   │   → logits      │       │   → tanh scalar │
   │   (one per      │       │   (predicted    │
   │   legal action) │       │   game outcome) │
   └─────────────────┘       └─────────────────┘
```

### Why Residual Networks?

Plain CNNs deeper than ~10 layers degrade in trainability — gradients
either vanish or explode, and the network becomes harder to optimize than
a shallower one. **Residual blocks** add a skip connection (`output =
F(x) + x`) that preserves a path for gradients to flow back through, which
makes very deep networks (40+ residual blocks in AlphaZero) stable to
train. This is the same architecture family as ImageNet's ResNet-50/152.

### Two Heads, Shared Trunk

Both heads sit on top of the same trunk, so the residual tower learns
representations useful for *both* "what move should I make?" (policy) and
"how good is this position?" (value). This shared learning is widely
believed to be one of the reasons AlphaZero generalizes so efficiently
across games.

### Sizes in Practice

| System            | Residual blocks | Filters/block | Parameters |
|-------------------|----------------:|--------------:|-----------:|
| AlphaZero (paper) | 19–39           | 256           | ~46–96 M   |
| Leela Chess Zero  | 10–40           | 64–384        | varies     |
| KataGo (b40c256)  | 40              | 256           | ~46 M      |
| dlshogi-large     | 15–20           | 192–256       | ~10–25 M   |

For comparison, ShogiDojo's HalfKAv2 NNUE is ~5 M parameters, with
**99.7%** of those concentrated in the sparse Feature Transformer that
runs *once per position* and then is incrementally updated. AlphaZero
networks are **~10–20× larger** and do dense convolutions on every
evaluation — the speed/accuracy tradeoff is fundamentally different.

---

## 5. Input Encoding for Shogi

The network input is a stack of 9×9 binary "planes," one per piece type and
condition. AlphaZero's own paper describes 119 input planes for Shogi
arranged as:

| Group                              | Planes | Notes                                              |
|------------------------------------|-------:|----------------------------------------------------|
| Own pieces, by type                |     14 | One plane per piece type incl. promoted (P, L, N, S, G, B, R, K, +P, +L, +N, +S, +B, +R) |
| Opponent pieces, by type           |     14 | Same                                                |
| Own hand piece counts              |      7 | One plane per droppable type (P, L, N, S, G, B, R) — value = count, broadcast across the whole 9×9 plane |
| Opponent hand piece counts         |      7 | Same                                                |
| Repetition counts                  |      3 | "Has this position occurred 1, 2, or 3 times?"      |
| **Per-position subtotal**          |     45 |                                                     |
| × History length (last 8 plies)    |    360 | The above 45 planes for each of the most recent 8 positions |
| Constant planes (color, move count, …) |  ~7 | Side-to-move indicator, total move count, no-progress count, etc. |
| **Total**                          |   ~119 |                                                     |

> Numbers above match the AlphaZero paper for Shogi. Open-source
> reimplementations (dlshogi, AobaZero) vary the exact plane count to suit
> their feature engineering preferences.

### Compared to HalfKAv2

| Aspect             | NNUE (HalfKAv2)                          | AlphaZero (planes)                         |
|--------------------|------------------------------------------|--------------------------------------------|
| Shape              | Sparse list of ~40 active feature indices | Dense 9×9×119 tensor                       |
| Total dim          | 19,638                                   | 9,639 (= 9×9×119) — but dense              |
| King handling      | Picks a "king bucket" weight slice        | Just another piece plane                   |
| Symmetry           | Explicit left-right file mirror           | None (the conv kernels learn symmetry)     |
| History            | Single position only                      | Last 8 plies stacked                       |

The AlphaZero encoding is conceptually simpler — there is no king bucket,
no thermometer encoding, no STM/NSTM split — but it is far more expensive
to consume because every evaluation processes a dense ~10K-element tensor
through dozens of convolutional layers.

### Side-to-Move Normalization

The AlphaZero paper, like most modern implementations, presents the board
*from the side-to-move's perspective*: when it is White's turn, the board
is flipped 180° so that the side-to-move's pieces always sit at the
"bottom." This is exactly the same idea as NNUE's STM perspective handling
— the network only ever has to learn one "point of view."

---

## 6. Move (Action) Encoding for Shogi

The policy head outputs one logit per *possible* action. To make this
finite and learnable, we map every Shogi move to a fixed action index.

The AlphaZero Shogi encoding uses **9 × 9 × 139 = 11,259 possible
actions**, decomposed as:

| Group                          | Actions per square | Total |
|--------------------------------|-------------------:|------:|
| Sliding moves (queen-like)     |       8 dirs × 8 distances | 64 |
| Knight moves (Shogi-specific)  |                  2 |     2 |
| Promoting variants of above    | (separate plane bit) |    + |
| Drops (piece-from-hand → square) |                7 |     7 |

In practice the policy is implemented as **9 × 9 × 139 logits**, where the
"from" square indexes the first two dimensions and the third dimension
selects the move type:

- 64 planes for sliding moves (8 directions × 8 distances)
- 2 planes for knight moves (Shogi knight has only two unique landing offsets)
- 64 + 2 planes again for the *promoting* variants of the above
- 7 planes for drops (one per droppable hand piece type)

A move is considered legal in two stages:

1. The geometric/rule-based legality check (handled by `shogi-rs` in our
   case — illegal moves are masked out *before* the softmax).
2. Among legal moves, the network's softmaxed logits give a probability
   distribution.

### Why Mask Illegal Moves?

The action space (~11K) is much larger than the legal-move count in any
single position (typically 30–80 legal moves in Shogi). Without masking,
the softmax would assign probability mass to illegal actions, which:

- Wastes capacity (the network would have to learn legality from scratch).
- Causes MCTS to consider impossible moves.
- Makes the softmax denominator dominated by illegal entries, distorting
  legal-move probabilities.

The standard trick is to set the logits of illegal moves to `-∞` (in
practice, a large negative number like `-1e9`) before applying softmax.

---

## 7. Monte Carlo Tree Search with PUCT

MCTS is what turns the network's raw output into actual play. AlphaZero
uses a specific MCTS variant called **PUCT** (Predictor + Upper
Confidence bound for Trees), which uses the network's policy as a prior.

### Tree Anatomy

Each node in the tree represents a position; each edge represents a legal
move. Edges store statistics that get refined with every simulation:

| Symbol | Stored on edge | Meaning                                        |
|--------|----------------|------------------------------------------------|
| `N`    | visit count    | How many simulations passed through this edge  |
| `W`    | total value    | Sum of value estimates from those simulations  |
| `Q`    | mean value     | `W / N`                                        |
| `P`    | prior          | Network's policy probability for this move     |

### One MCTS Simulation

A single simulation consists of four phases:

```
  ┌───────────────────────────────────────────────────────────────┐
  │ 1. SELECT                                                      │
  │   Starting from the root, descend the tree by repeatedly       │
  │   choosing the action a that maximizes:                         │
  │                                                                 │
  │           U(s, a) = Q(s, a) + c_puct · P(s, a) · √ΣN / (1+N)    │
  │                     └─exploit─┘   └────────explore─────────┘    │
  │                                                                 │
  │   Stop when we reach a leaf (a node not yet expanded).          │
  │                                                                 │
  │ 2. EXPAND                                                       │
  │   Run the network on the leaf position:                         │
  │      π_net, v = network(state)                                  │
  │   Add child edges for every legal move, initialized with        │
  │   N=0, W=0, Q=0, P=π_net[a].                                    │
  │                                                                 │
  │ 3. EVALUATE                                                     │
  │   Use v (the network's value estimate) as the leaf's value.     │
  │   No Monte Carlo rollouts — that was AlphaGo, not AlphaZero.    │
  │                                                                 │
  │ 4. BACKUP                                                       │
  │   Walk back up the path you came from, and for each edge:       │
  │      N ← N + 1                                                  │
  │      W ← W + v_for_that_perspective                             │
  │      Q ← W / N                                                  │
  │   Note: v flips sign at every ply because the perspective       │
  │   alternates between players.                                   │
  └───────────────────────────────────────────────────────────────┘
```

A typical AlphaZero search performs **800 simulations** per move during
training self-play, and sometimes thousands more during evaluation
matches. Each simulation is one network forward pass plus tree bookkeeping.

### The PUCT Formula in Plain Words

```
U(s, a) = Q(s, a) + c_puct · P(s, a) · √(Σ_b N(s, b)) / (1 + N(s, a))
```

- `Q(s, a)` — exploit: pick moves we have already seen do well.
- `c_puct · P(s, a) · √N / (1 + N)` — explore:
  - High when the network's prior `P` is high, biasing search toward
    plausible moves.
  - High when the move has been visited few times relative to its siblings.
  - The `c_puct` constant trades exploration vs exploitation
    (~1.0 to ~4.0 in practice).

Compared to alpha-beta search, PUCT MCTS is **best-first**: it spends
visits where the network and prior history say they are likely to matter,
not on every branch up to a fixed depth.

### Dirichlet Noise at the Root

To force exploration during self-play, AlphaZero adds Dirichlet noise to
the *root's* prior policy at every move:

```
P(root, a) ← (1 − ε) · P(root, a) + ε · Dir(α)
```

- `ε ≈ 0.25` (so 25% of the prior comes from noise)
- `α` is set so the noise distribution is roughly flat over legal moves.
  For Shogi, `α ≈ 0.15` is typical (chess uses 0.3, Go uses 0.03 — smaller
  values for larger move spaces).

Without this noise, the same self-play game would tend to repeat from
similar openings, depriving training of position diversity.

### Move Selection from Visit Counts

After all simulations are complete, the move played is sampled from a
*temperature-controlled distribution* over visit counts:

```
π(a) ∝ N(root, a)^(1/τ)
```

- **Early in the game** (typically the first 30 plies): `τ = 1`, so play
  proportionally to visit counts. This produces diverse openings.
- **Later in the game**: `τ → 0`, which reduces to `argmax N(root, a)`.
  Greedy play in critical positions.

The `π` distribution itself (the visit-count distribution at temperature
1, *before* the optional argmax) is the training target for the policy
head — see [Loss Function](#9-loss-function).

---

## 8. Self-Play and the Training Loop

Self-play is the data source. There is no human game database, no
teacher engine, no curated dataset.

```
While training:
    while replay_buffer.size < min_buffer_size:
        play_one_self_play_game()

    for step in range(steps_per_epoch):
        batch = replay_buffer.sample(batch_size)
        loss = compute_loss(network, batch)
        optimizer.step(loss)

    if step % checkpoint_every == 0:
        save_checkpoint()
        if evaluation_match(latest, current_best).won:
            current_best = latest
```

Open-source projects (Leela Chess Zero, dlshogi, AobaZero) use a
distributed variant: many workers play self-play games and upload them to
a central server, while a separate trainer continuously samples from the
combined buffer.

### One Self-Play Game

```
state = initial_position()
game_history = []

while not state.terminal:
    # Run MCTS with the latest network
    visit_counts = mcts.search(state, num_simulations=800)

    # The training target for the policy at this state
    pi = visit_counts ** (1 / temperature)
    pi /= pi.sum()

    # Record (state, pi); we will fill in z later
    game_history.append((state, pi))

    # Sample the actual move played
    action = sample_from(pi if move_count < 30 else argmax)
    state = state.apply(action)

# Backfill the terminal value into every position of the game
z = state.outcome  # +1 if the side-to-move at the terminal lost, etc.
for state_t, pi_t in game_history:
    perspective = state_t.side_to_move
    z_for_t = z if perspective == terminal_perspective else -z
    replay_buffer.add(state_t, pi_t, z_for_t)
```

### What Goes Into the Replay Buffer?

Each training example is a tuple **(state, π, z)**:

- `state` — the board representation (the 9×9×119 input planes).
- `π` — the MCTS visit-count distribution at that state. This is the
  *improved* policy that we want the network to imitate.
- `z` — the eventual game outcome from this state's side-to-move's
  perspective: `+1` win, `0` draw, `−1` loss.

> Note the asymmetry with NNUE training data: NNUE uses (state, score,
> result) where the score comes from a *separate evaluator* (engine
> search). AlphaZero uses (state, visit-counts, result) where the
> "score-like" target is generated by the same network's own MCTS. This
> is the bootstrap that makes self-play possible.

---

## 9. Loss Function

The combined loss has three terms:

```
L = (z − v)²              ← value loss (MSE against terminal outcome)
  − π · log(p_net)        ← policy loss (cross-entropy vs MCTS visits)
  + c · ||θ||²            ← L2 weight regularization
```

Where:

- `v` is the value head's scalar prediction in `[−1, +1]`.
- `p_net` is the policy head's softmax over legal moves.
- `θ` is all network parameters.
- `c` is typically `1e-4`.

### Why MSE for Value, Cross-Entropy for Policy?

- **Value (MSE)**: We want the prediction `v` to match the actual game
  outcome `z`. MSE is the natural choice for a continuous target in
  `[−1, +1]`. Stockfish-style score-target training uses sigmoid + MSE in
  a probability space; AlphaZero uses raw `tanh(v) ≈ z` MSE for
  simplicity.
- **Policy (cross-entropy)**: We want `p_net` to match the *distribution*
  `π`, not just its argmax. Cross-entropy is the standard loss for
  matching distributions over discrete classes. Critically, the target
  `π` is the *improved* policy from MCTS, not the raw network policy
  from the last iteration — this is what makes the training loop a
  bootstrap rather than a fixed point.

### Compared to NNUE

| Aspect          | NNUE Training                                       | AlphaZero Training                                |
|-----------------|-----------------------------------------------------|---------------------------------------------------|
| Targets         | Engine score (from external engine), game result    | MCTS visit distribution (self-generated), result  |
| Score loss      | `MSE(σ(v_net), σ(score))` — sigmoid space            | `MSE(v_net, z)` — direct outcome                  |
| Policy loss     | Not applicable — NNUE has no policy head            | `−π · log(p_net)` — cross-entropy                  |
| Data source     | Pre-generated SFEN files (off-line)                  | Live replay buffer fed by self-play (on-line)     |
| Lambda weights  | `λ_score=1.0`, `λ_result=0.5`                        | All terms equal weight in original paper           |

---

## 10. Comparison with NNUE

A side-by-side summary of design tradeoffs, with practical numbers where
possible. Numbers are **order-of-magnitude estimates** drawn from
published systems and open-source engines, not benchmarks of any
ShogiDojo implementation.

| Dimension                  | NNUE (HalfKAv2)                       | AlphaZero-style                                |
|---------------------------|---------------------------------------|------------------------------------------------|
| Network size              | ~5 M params                           | ~10–100 M params                               |
| Per-eval cost (CPU)       | ~100 ns (incremental)                 | ~10–100 ms (full forward pass)                 |
| Eval count per move       | 10⁶–10⁷ (alpha-beta)                  | 10² –10³ (MCTS)                                |
| Move-time floor           | < 1 ms feasible                       | ≥ 100 ms typical                               |
| Hardware                  | CPU-friendly; GPU optional             | GPU/TPU strongly preferred                     |
| Training data source      | External engine search (off-line)      | Self-play with live network (on-line)          |
| Training compute          | 1 GPU-day for usable model             | 100s of GPU-days for competitive model         |
| Code complexity           | Static eval + classical search         | MCTS + replay buffer + distributed self-play   |
| Where parameters live     | 99.7% in sparse FT (lookup-only)       | Distributed across deep conv layers            |
| Inference engine fit      | Designed for CPU game engines          | Designed for GPU-driven search                 |

### Why both paradigms exist

- NNUE wins on **deployment economics**: it runs on a phone, a laptop, or a
  CPU server, at sub-millisecond move times. A user-facing Shogi app can
  ship NNUE models without a GPU dependency.
- AlphaZero wins on **research economics**: it learns from scratch with no
  curated dataset and with minimal game-specific engineering. New games
  or new variants (Crazyhouse, Atomic, Antichess) can be tackled with the
  same code.
- Modern strongest engines (Stockfish for chess, dlshogi for Shogi) are
  hybrids: they use NNUE-style evaluation inside an MCTS *or* alpha-beta
  search informed by AlphaZero-style training data. Surveying both
  paradigms is preparation for understanding such systems.

---

## 11. Glossary

| Term                       | Definition |
|----------------------------|------------|
| **AlphaZero**              | DeepMind's 2017 system that learns chess, Shogi, and Go from scratch via self-play with a single residual network and PUCT MCTS. Often used loosely for the broader family of similar systems. |
| **Backup (MCTS)**          | The fourth phase of an MCTS simulation: walk back from the leaf to the root, updating visit counts and value sums on every edge along the path. |
| **c_puct**                 | The exploration constant in the PUCT formula. Larger values bias search toward exploring less-visited moves with high prior. |
| **Dirichlet noise**        | Random perturbation added to the root's prior policy at every self-play move to force exploration. Controlled by `ε` (mixture weight) and `α` (concentration). |
| **MCTS**                   | Monte Carlo Tree Search. A best-first search algorithm that builds a tree of positions and uses simulated playouts (or, in AlphaZero, network value estimates) to guide expansion. |
| **Policy head**            | The output of the network that produces a probability over legal moves. Trained to imitate MCTS visit counts. |
| **Prior (P)**              | The network's policy probability for a move, used as the initial bias term in PUCT before any simulations have visited that edge. |
| **PUCT**                   | "Predictor + UCT" — the MCTS selection formula used by AlphaZero. Combines a value-exploitation term `Q` with an exploration term that scales with the network's prior. |
| **Replay buffer**          | The pool of recent (state, π, z) tuples generated by self-play. The trainer samples mini-batches from this buffer. |
| **Residual block**         | A two-conv-layer block with a skip connection (`output = F(x) + x`) that enables training very deep networks without gradient pathology. |
| **Self-play**              | The technique of generating training data by having the current network play games against itself, with MCTS at every move. |
| **Temperature (τ)**        | A parameter that controls how greedy the move selection is from the visit-count distribution. `τ = 1` samples proportionally; `τ → 0` becomes argmax. |
| **Value head**             | The output of the network that produces a single scalar in `[−1, +1]` predicting the eventual game outcome from the current state. |
| **Visit count (N)**        | The number of MCTS simulations that have passed through a given edge. The visit-count distribution at the root is the policy training target. |
| **z**                      | The terminal game outcome (`+1`, `0`, or `−1`) viewed from a particular state's side-to-move's perspective. The training target for the value head. |
| **π**                      | The MCTS-derived policy at a state — the visit-count distribution (optionally raised to a temperature). The training target for the policy head. |

---

## 12. References

### Primary sources

- D. Silver, T. Hubert, J. Schrittwieser, et al. *"Mastering chess and
  shogi by self-play with a general reinforcement learning algorithm,"*
  arXiv:1712.01815 (2017). The AlphaZero paper.
- D. Silver, J. Schrittwieser, K. Simonyan, et al. *"Mastering the game
  of Go without human knowledge,"* Nature 550, 354–359 (2017). The
  AlphaGo Zero paper, where the policy/value-with-MCTS recipe was first
  introduced.
- D. Silver et al., *"A general reinforcement learning algorithm that
  masters chess, shogi, and Go through self-play,"* Science 362,
  1140–1144 (2018). The peer-reviewed Science version of the AlphaZero
  result, with additional analysis.

### Open-source implementations worth reading

- [Leela Chess Zero](https://lczero.org/) — the most widely used
  open-source AlphaZero-style chess engine. Excellent codebase to read
  for distributed self-play and engineering details.
- [KataGo](https://github.com/lightvector/KataGo) — a Go engine that
  refines AlphaZero with several clever ideas (auxiliary policy targets,
  SAI-style score prediction, dynamic komi).
- [dlshogi](https://github.com/TadaoYamaoka/DeepLearningShogi) — Shogi
  engine in this family, the closest reference for Shogi-specific
  encoding and engineering choices.
- [AobaZero](https://github.com/kobanium/aobazero) — Japanese-led
  open-source AlphaZero for Shogi project; useful for cross-checking
  Shogi-specific input planes and move encoding.

### Tutorials and pedagogical material

- *"A Simple Alpha(Go) Zero Tutorial"* — a popular blog-post + minimal
  Connect4 implementation that captures the essential algorithm without
  game-specific complexity.
- The AlphaZero paper's pseudocode appendix (in arXiv:1712.01815) is
  short and unambiguous; reading it directly is highly recommended once
  the high-level picture from this document is clear.

### Cross-references within ShogiDojo

- [`doc/nnue/NNUE_Reference.md`](../nnue/NNUE_Reference.md) — the
  parallel reference for the NNUE paradigm.
- [`doc/nnue/Model_Training_Reference.md`](../nnue/Model_Training_Reference.md)
  — training methodology for NNUE, useful for contrast with the
  AlphaZero training loop described in §8 above.
- [`doc/alphazero/AlphaZero_Training_Reference.md`](AlphaZero_Training_Reference.md)
  — deeper treatment of training methodology, replay-buffer dynamics,
  hyperparameter ranges (Dirichlet noise, c_puct, temperature), the
  training loop in detail, evaluation/promotion gating, distributed
  self-play, resource estimates, common failure modes, and a side-by-side
  comparison with NNUE training.
