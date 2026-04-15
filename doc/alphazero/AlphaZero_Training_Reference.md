# AlphaZero Training Reference

This document explains the *training methodology* of AlphaZero-style
systems — what the training loop actually does, how the hyperparameters
interact, what fails and how to recognize it, and what realistic compute
budgets look like.

It is the AlphaZero counterpart to
[`doc/nnue/Model_Training_Reference.md`](../nnue/Model_Training_Reference.md)
and assumes you have already read
[`doc/alphazero/AlphaZero_Reference.md`](AlphaZero_Reference.md), which
introduces the architecture, MCTS, the high-level loop, and the
vocabulary. Where useful, this document points back at the parallel NNUE
section so the contrast is explicit.

> **Status — research survey in progress.** ShogiDojo does not yet
> contain an AlphaZero-style implementation. Numbers in this document
> are drawn from published systems and open-source projects (Leela
> Chess Zero, KataGo, dlshogi, AobaZero, the original AlphaZero paper).
> They are order-of-magnitude estimates, not benchmarks of any local
> code. Anything labeled "default" reflects what the literature
> converges on, and is intended as a starting point for our own
> experiments — not as authoritative.

---

## Table of Contents

1. [Self-Play as the Data Source](#1-self-play-as-the-data-source)
2. [The Replay Buffer](#2-the-replay-buffer)
3. [Loss Function in Detail](#3-loss-function-in-detail)
4. [Hyperparameters](#4-hyperparameters)
5. [The Training Loop, Step by Step](#5-the-training-loop-step-by-step)
6. [Evaluation Matches and Network Promotion](#6-evaluation-matches-and-network-promotion)
7. [Distributed Self-Play](#7-distributed-self-play)
8. [Resource Estimates](#8-resource-estimates)
9. [Common Failure Modes](#9-common-failure-modes)
10. [Improvement Strategy](#10-improvement-strategy)
11. [Comparison with NNUE Training](#11-comparison-with-nnue-training)
12. [Realistic Expectations for a Research Survey](#12-realistic-expectations-for-a-research-survey)

---

## 1. Self-Play as the Data Source

### The Bootstrap

Unlike NNUE training — where data comes from an *external* engine
search and is fixed before training starts — AlphaZero training data is
produced *by the same system that is being trained*. There is no
external teacher. The bootstrap works because of one key property of
MCTS:

> **MCTS run with a network produces a policy that is, in expectation,
> better than the network's raw policy.**

This is the "policy improvement operator" claim from the original
AlphaGo Zero paper. Even a randomly initialized network, run inside
MCTS with enough simulations, produces visit-count distributions that
are noticeably more informed than the raw output. Training the network
to imitate those visit counts pulls its raw policy *toward* the
improved policy. With each iteration, MCTS-on-the-better-network
produces still-better visit counts, and so on.

### The Practical Consequence

If you only ever trained on raw network outputs, the network would have
nothing to learn from — its target would be its own current state, and
loss would be zero from epoch one. The MCTS step is what creates a
*signal*: there is a difference between what the network would say
unaided and what it produces after thinking with search. That difference
is the gradient that improves the network.

A useful mental model: **MCTS is a slow, expensive teacher; training
distills its output back into the fast, cheap student**. The student
gets faster every iteration. Eventually, the student approaches the
quality of the teacher *at one MCTS-simulation count*, at which point
you can either increase the teacher's simulation budget (push the
ceiling up) or stop.

### Side-to-Move Convention

As with NNUE training (see
[`Model_Training_Reference.md` §1](../nnue/Model_Training_Reference.md#perspective-handling-during-training)),
AlphaZero training uses a side-to-move-relative representation
throughout:

- The board planes are flipped 180° when it is White to move, so the
  network always sees its own pieces "at the bottom."
- The value target `z` is `+1` if the side-to-move at that position
  ultimately won, `−1` if they lost, `0` for draw.
- The value head outputs `tanh`-bounded scalars, also from the
  side-to-move's perspective.

This means the network **never has to learn that Black and White are
different roles**. It only has to learn good Shogi.

---

## 2. The Replay Buffer

The replay buffer is the bridge between self-play and training. It
holds a sliding window of the most recent `(state, π, z)` tuples.

### Why a Buffer Instead of Direct Streaming?

Two reasons:

1. **Decoupling**: Self-play and training run at different speeds and
   often on different hardware. The buffer lets each side proceed at
   its own rate without blocking.
2. **Temporal mixing**: Mini-batches sampled from a buffer mix
   positions from many different games, which gives gradient updates
   lower variance than streaming positions in game order would. Game
   order is highly correlated (consecutive positions in one game share
   most pieces); buffered sampling breaks that correlation.

### Buffer Size

| System (open-source)  | Buffer size                          |
|-----------------------|--------------------------------------|
| AlphaZero paper       | ~500,000 most recent self-play games |
| Leela Chess Zero      | ~1M positions, sliding window         |
| KataGo                | adaptive — sized by recent generations |
| dlshogi               | ~1–2M positions                      |

For a laptop-scale didactic implementation, a buffer of **~10K–100K
positions** is sufficient to demonstrate the idea without OOM-ing.

### Age-Out Policy

The replay buffer is FIFO: when full, the oldest position is evicted as
each new position arrives. This matters because **stale positions —
generated by much earlier (weaker) versions of the network — drag
training in the wrong direction.** If the network has improved
significantly since a position was generated, the MCTS distribution `π`
recorded on that position no longer reflects what the *current*
network's MCTS would produce. The training target is misleading.

A reasonable rule of thumb: the buffer should hold roughly
**1–4 generations'** worth of self-play data, where a "generation" is
the period between checkpoints.

### Sampling

Mini-batches are sampled **uniformly at random from the buffer** at
training time. Some implementations weight more-recent samples slightly
higher, but the simple uniform sample is the AlphaZero default and is
hard to beat.

Per-batch, all the `(state, π, z)` tuples in the batch are fed through
the network and their loss contributions averaged. Standard mini-batch
SGD/Adam.

---

## 3. Loss Function in Detail

The combined loss has three terms:

```
L = (z − v_θ(s))²                    ← value loss
  − π(a|s) · log p_θ(a|s)            ← policy loss
  + c · ||θ||²                       ← L2 regularization
```

### Value Loss — MSE Against the Game Outcome

```
L_value = (z − v)²
```

- `v` is the value head's prediction in `[−1, +1]` (after tanh).
- `z ∈ {−1, 0, +1}` is the eventual game outcome from this state's
  side-to-move's perspective.
- Standard squared-error regression; nothing exotic.

> **Compared to NNUE's score loss**: NNUE uses
> `MSE(σ(v_predicted), σ(v_target))` in *probability space*, with
> `score_scaling=600` to map centipawns into a sensible sigmoid range.
> AlphaZero compares `v` and `z` directly because both already live in
> `[−1, +1]` — the network is predicting outcome probability minus 0.5,
> doubled, with no need for further scaling.

### Policy Loss — Cross-Entropy Against MCTS Visit Counts

```
L_policy = − Σ_a π(a|s) · log p_θ(a|s)
```

- `π(a|s)` is the MCTS-derived target distribution at state `s` —
  roughly proportional to visit counts, optionally raised to a
  temperature.
- `p_θ(a|s)` is the network's softmax over legal moves.
- This is the standard cross-entropy loss for matching two
  distributions over discrete classes.

Critically, **the policy target is the MCTS-improved policy, not the
network's previous policy**. This is what gives the loss a non-zero
gradient over time: without MCTS, the target would equal the prediction
on every iteration.

### L2 Weight Regularization

```
L_reg = c · ||θ||²    (c ≈ 1e-4)
```

A standard L2 penalty on all network parameters. Prevents weights from
growing unbounded over many millions of training steps. Typical values
are between `1e-5` and `1e-4`.

### Balancing the Terms

In the original AlphaZero paper, all three terms are summed with equal
weight. Some open-source implementations introduce a **value loss
scale** (e.g., `1.0` for policy, `1.5` for value) to compensate for the
fact that policy gradients are spread across many output classes while
value gradients concentrate on a single scalar. This is a tunable knob,
but the default-equal weighting is a fine starting point.

> **Compared to NNUE**: NNUE uses `λ_score=1.0`, `λ_result=0.5` —
> *score* (engine evaluation) is weighted higher because it carries
> finer-grained positional information, while *result* (final outcome)
> is noisier. AlphaZero has no engine-score signal at all; the only
> training targets are the MCTS visit distribution and the game
> outcome, both of which are self-generated.

---

## 4. Hyperparameters

### MCTS Hyperparameters

These control the *teacher* — the data-generation side of the loop.

#### `num_simulations` — MCTS playouts per move

The single most important compute knob. More simulations = stronger
teacher = better training data, but linearly slower self-play.

| Setting               | Use case                                    |
|-----------------------|---------------------------------------------|
| 50–100                | Sanity-check / didactic implementations      |
| 200–400               | Laptop-scale experiments                     |
| 800                   | AlphaZero paper default for Shogi/chess      |
| 1,600                 | Leela Chess Zero training                    |
| 3,000+                | Evaluation matches (not training)            |

For research-survey purposes, 50–200 is enough to *see the system
work*. Anything beyond that is for actually trying to make a strong
network.

#### `c_puct` — exploration constant

Controls how aggressively MCTS explores low-prior moves.

```
U(s, a) = Q(s, a) + c_puct · P(s, a) · √(Σ_b N(s, b)) / (1 + N(s, a))
```

Typical values:

- `c_puct = 1.0` — original AlphaZero paper, conservative
- `c_puct = 1.5–2.5` — most open-source implementations
- `c_puct = 4.0+` — more aggressive exploration; useful early in
  training when the network's prior is bad

Some systems use a *dynamic* `c_puct` that grows logarithmically with
total visits at the parent node — this lets exploration scale up as
search budget grows. The two main variants are constant `c_puct`
(simpler) and `c_puct(N) = c_init + log((1 + N + c_base) / c_base)`
(LC0, KataGo).

#### Dirichlet noise — `(α, ε)`

Added to the root's prior at every self-play move to force exploration.

```
P_root(a) ← (1 − ε) · P_θ(a) + ε · Dir(α)
```

| Game  | α     | ε    | Why                                                |
|-------|-------|------|----------------------------------------------------|
| Chess | 0.30  | 0.25 | ~20–40 legal moves; α ≈ 1/avg_legal_moves          |
| Shogi | 0.15  | 0.25 | ~30–80 legal moves (drops widen the move space)    |
| Go    | 0.03  | 0.25 | ~250+ legal moves (very flat target distribution)  |

`ε = 0.25` means 25% of the root prior comes from noise — enough to
push search into otherwise-ignored moves, not so much that it
overwhelms the network's signal.

Noise is applied **only at the root** of the search tree, and **only
during self-play**. Evaluation matches use the network's clean prior.

#### Temperature schedule — `τ(move)`

Controls how greedily the actual move played is sampled from the
visit-count distribution.

```
P(play a) ∝ N(root, a)^(1/τ)
```

The standard schedule:

```
τ = 1.0   for move ≤ 30   (sample proportionally to visits — diversity)
τ → 0     for move > 30   (argmax — best move only)
```

The crossover threshold (30 plies in the AlphaZero paper) is a
hyperparameter; values between 15 and 50 are reasonable. Lower
crossover means "be greedy sooner," which produces faster but less
diverse training games.

The training target `π` is the visit distribution at `τ = 1`
*regardless* of which `τ` was used to pick the actual move. This is
important: even when we play greedily, the policy target retains the
full distributional information.

### Optimizer Hyperparameters

These control the *student* — the network update side of the loop.

| Hyperparameter   | AlphaZero paper          | Common alternatives                         |
|------------------|--------------------------|---------------------------------------------|
| Optimizer        | SGD with momentum (0.9)  | Adam (LCZero), RMSProp                      |
| Learning rate    | 0.2 → 0.02 → 0.002 → 0.0002 (stepwise) | 1e-3 → 1e-4 cosine (Adam-style) |
| Weight decay     | 1e-4                     | 1e-5 to 1e-4                                |
| Batch size       | 4,096                    | 256–8,192 depending on GPU memory           |
| Gradient clipping| Optional; max-norm 1–10  | Often omitted with Adam                     |

The original paper used SGD-with-momentum and a hand-stepped LR
schedule. Open-source projects more commonly use Adam, which is more
forgiving on the LR setting at the cost of slightly worse final
generalization.

For a laptop-scale didactic run, **Adam at `lr = 1e-3`** with no
schedule is fine. The point of a research survey is not to wring out
the last 10% of strength.

---

## 5. The Training Loop, Step by Step

```python
# Pseudo-code, intentionally readable rather than fast.

network          = Network()           # randomly initialized
best_network     = copy(network)        # for evaluation matches
replay_buffer    = ReplayBuffer(max_size=100_000)
optimizer        = Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)

for iteration in range(num_iterations):

    # --- 1. Self-play: produce data with the latest weights ---
    for _ in range(games_per_iteration):
        game = play_self_play_game(
            network=network,
            num_simulations=200,
            c_puct=1.5,
            dirichlet_alpha=0.15,
            dirichlet_epsilon=0.25,
            temperature_threshold=30,
        )
        for state, pi, z in game.training_examples():
            replay_buffer.add(state, pi, z)

    # --- 2. Train: sample mini-batches from the buffer ---
    for _ in range(training_steps_per_iteration):
        batch = replay_buffer.sample(batch_size=256)
        loss = compute_loss(network, batch)        # value + policy + L2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- 3. Evaluate: gate the new network against the best-so-far ---
    if iteration % evaluation_every == 0:
        result = run_match(
            network_a=network,
            network_b=best_network,
            num_games=40,
            num_simulations=400,        # higher than self-play
        )
        if result.win_rate(network) > 0.55:
            best_network = copy(network)
            save_checkpoint(network, iteration, label="best")

        save_checkpoint(network, iteration, label="latest")
```

A few practical notes:

- **Self-play and training can interleave or run in parallel.** The
  pseudo-code above runs them sequentially for clarity. In open-source
  implementations, self-play is usually a separate process (or many
  processes on a cluster) that pushes data into a shared buffer, and
  the trainer continuously samples and steps in another process.
- **`evaluation_every` is typically large** (every 10–100 iterations),
  because evaluation matches are expensive and the network changes
  slowly over a single iteration.
- **Both `latest` and `best` checkpoints are kept.** Self-play uses the
  `latest` (so the data reflects current capability), but for
  evaluation against external systems you usually want `best`.

---

## 6. Evaluation Matches and Network Promotion

### Why Gate Network Promotion?

Modern AlphaZero implementations differ from the original paper here.
The paper trained a single network continuously and never "promoted"
explicitly. Open-source projects (LCZero in particular) found that
**not every training step produces an improvement** — sometimes a
batch causes a regression, and using a worse network for self-play
poisons subsequent data.

The fix is to require an evaluation match before letting a new network
generate self-play data. Only "promote" the new network if it wins by
a clear margin (typically ≥55% win rate in 40+ games against the
current best).

### Match Configuration

| Parameter            | Typical value | Notes                                                  |
|----------------------|--------------:|--------------------------------------------------------|
| Number of games      |     40–400    | More games = more confidence in result                  |
| MCTS simulations     |     400+      | Higher than self-play; produces sharper play            |
| Temperature          |      0.1      | Near-greedy for accurate strength comparison            |
| Dirichlet noise      |      off      | No exploration during evaluation                        |
| Color swapping       |      yes      | Each pair of games swaps which network plays Black      |
| Promotion threshold  |     ≥55% wr   | Usually defined excluding draws or weighting them 0.5   |

This is conceptually identical to the NNUE `eval_match` tool
([Model_Training_Reference.md §11](../nnue/Model_Training_Reference.md#measuring-model-strength))
— color-swapped pairs, score reporting, Elo difference estimation. We
will reuse the same statistics framework when implementing AlphaZero
evaluation.

### How Many Games for Confidence?

(Same calculus as NNUE evaluation matches.)

| Games | ±Elo confidence (ish) | Use case                                     |
|-------|----------------------:|----------------------------------------------|
| 40    |     ±100             | Promotion gate during training                |
| 200   |     ±50              | Routine regression check                      |
| 1,000 |     ±20              | Confirming small claimed improvements         |

For training-loop promotion, 40 games is the LCZero default — the
threshold of ≥55% on 40 games already implies a significantly stronger
network with reasonable confidence.

---

## 7. Distributed Self-Play

For a research survey we will run everything in a single process. But
because most of the literature describes distributed setups, it is
worth understanding the architecture you will see referenced in
papers and codebases.

```
                          ┌────────────────────────┐
                          │   Trainer (single GPU) │
                          │                         │
                          │ pull mini-batches ◄─────┼──┐
                          │ apply gradients         │  │
                          │ push new weights ───────┼──┼──┐
                          └────────────────────────┘  │  │
                                                       │  │
                          ┌────────────────────────┐  │  │
                          │  Replay Buffer         │  │  │
                          │  (network-attached     │◄─┘  │
                          │   server, e.g. Redis)  │     │
                          └────────────────────────┘     │
                                       ▲                  │
                  push training         │                  │
                  examples              │                  │
                                       │                  │
              ┌────────┬────────┬──────┴──────┬────────┐  │
              │        │        │             │        │  │
       ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐    │  │
       │worker 1 │ │worker 2│ │worker 3│ │worker N│ ◄──┘──┘
       │ (GPU)   │ │ (GPU)  │ │ (GPU)  │ │ (GPU)  │    pull latest weights
       └─────────┘ └────────┘ └────────┘ └────────┘
```

Each self-play worker:

1. Periodically pulls the **latest network weights**.
2. Plays self-play games using its local network copy.
3. Pushes finished games' `(state, π, z)` tuples to the central
   replay buffer.

The trainer:

1. Continuously samples mini-batches from the buffer.
2. Applies gradient updates.
3. Periodically publishes new weights for workers to pull.

This is **off-policy training**: workers may be running with weights
that lag the latest by some amount (minutes or more in distributed
setups). This is generally fine — the policy doesn't change
catastrophically over a few minutes — but extreme staleness (hours+)
can hurt.

For a single-machine didactic implementation, this whole architecture
collapses to a single process that alternates between self-play and
training. The same loop, just without the network-attached buffer and
weight publishing.

---

## 8. Resource Estimates

Order-of-magnitude numbers for context. The actual number depends on
network size, MCTS budget, GPU model, and engineering quality.

### To Reach "Stronger Than PST Material Eval"

A sanity-check threshold; comparable to a freshly-trained NNUE first
iteration.

- **Network**: ~1M params (4 residual blocks × 64 filters)
- **Self-play**: ~10K games, each ~150 plies with 100 MCTS sims
- **Training**: ~50K mini-batch steps
- **Compute**: ~1 GPU-day on a consumer GPU

### To Reach "Reasonable Amateur Strength"

Comparable to a small NNUE trained on 1M positions.

- **Network**: ~10M params (10 residual blocks × 128 filters)
- **Self-play**: ~1M games
- **Training**: ~1M mini-batch steps
- **Compute**: ~30 GPU-days on a consumer GPU

### To Reach "Top Open-Source Engine Strength"

Comparable to a competitive NNUE trained on 1B+ positions plus
extensive self-play iteration.

- **Network**: ~50M params (20+ residual blocks × 256 filters)
- **Self-play**: ~10M+ games over months
- **Compute**: ~100s of GPU-years (open-source projects spread this
  across volunteer compute, e.g. LCZero's ~1000-contributor network).

For comparison, the original AlphaZero paper trained for **9 hours on
5,000 first-generation TPUs** to surpass Elmo at Shogi. That's roughly
~45,000 TPU-hours — equivalent to multiple GPU-years of consumer
hardware. The dominant cost is self-play, not training.

> **Implication for ShogiDojo's research survey**: We are not going to
> approach competitive strength. The goal is to have a working
> reference implementation that the team can run, read, and modify.
> Targeting "stronger than random play" on a small board or a tiny
> network is a reasonable success criterion.

---

## 9. Common Failure Modes

These are mistakes that show up *during training*, before any
evaluation match. They are easier to fix once you can recognize the
symptom.

### 9.1 Value Collapse

**Symptom**: After many iterations, the value head outputs cluster
near `0` for nearly all positions. The network has learned to predict
"draw" universally because draws are common and predicting `0` minimizes
MSE on a noisy outcome signal.

**Cause**: Insufficient self-play exploration; not enough decisive
games; or value loss weight too low.

**Fix**: Increase Dirichlet noise (`ε` from 0.25 → 0.4 temporarily) to
generate more decisive games. Confirm by checking the win/draw/loss
ratio of recent self-play games — aim for ≤30% draws early in training.

### 9.2 Policy Collapse

**Symptom**: The policy head concentrates ~all probability on a small
number of moves regardless of position. MCTS spends almost all visits
on those moves.

**Cause**: Network output is overfitting to a few common training
positions; or the learning rate is too high and the softmax has
saturated.

**Fix**: Reduce learning rate; increase weight decay; verify the
replay buffer contains diverse positions (sample some manually and
inspect).

### 9.3 The Stale Checkpoint Problem

**Symptom**: Self-play games look reasonable, but evaluation matches
against the previous network show no improvement or even regression.

**Cause**: Workers are using significantly older weights than the
latest checkpoint, so self-play data reflects an older (weaker)
distribution than the current model.

**Fix**: In a distributed setup, reduce the weight-pull interval. In a
single-machine setup, this should not occur — verify that self-play
and training share the same in-memory network.

### 9.4 Replay Buffer Staleness

**Symptom**: Loss decreases steadily but evaluation matches plateau
early.

**Cause**: Buffer is too large relative to training rate, so old
positions (generated by much weaker networks) make up most of every
mini-batch. The network is fitting the average of many older
distributions, not the current improved policy.

**Fix**: Decrease buffer size, or increase self-play generation rate
relative to training rate. As a rule of thumb, **the buffer should
turn over every few generations** — if positions sit there for >10
checkpoints' worth of iterations, they are stale.

### 9.5 NaN/Inf Loss

**Symptom**: Loss spikes to NaN partway through training.

**Cause**: Most often, an illegal action receiving non-zero policy
mass after softmax (because masking was forgotten) and contributing
`log(0)` to cross-entropy. Or unstable gradients from a too-high
learning rate.

**Fix**: Verify legal-move masking is applied **before** softmax (set
illegal logits to a large negative number). Add gradient clipping
(max-norm 1.0) if not already present.

### 9.6 Self-Play Hangs / Infinite Games

**Symptom**: A self-play game runs forever without terminating.

**Cause**: Sennichite (Shogi 4-fold repetition) detection or move
limits not implemented. AlphaZero-style search is happy to repeat
positions indefinitely if both sides see them as drawish.

**Fix**: Enforce a hard ply limit (e.g. 512) and treat any game
exceeding it as a draw. Implement repetition detection per Shogi
rules.

---

## 10. Improvement Strategy

The same hierarchy as
[NNUE training §10](../nnue/Model_Training_Reference.md#improvement-strategy):
**data quality dominates everything else** until it stops being the
bottleneck.

### 10.1 More Self-Play (Biggest Lever)

The single most reliable way to improve an AlphaZero network is
**more self-play games**, generated by the latest network. Doubling
the buffer turnover rate is roughly equivalent to doubling the
network's training budget.

If self-play is the bottleneck (it almost always is on small
hardware): reduce `num_simulations` per move (200 → 100) so games
finish faster, *or* shrink the network so each MCTS evaluation is
cheaper, *or* both.

### 10.2 Stronger Teacher (More MCTS Simulations)

Once self-play volume is no longer a constraint, increase
`num_simulations` to make the teacher stronger. This produces
higher-quality `π` targets and pulls the network up faster per
training step.

The tradeoff: at 800 simulations, each game takes ~8× longer than at
100 simulations. For research-survey purposes, the sweet spot is the
*highest* simulation count that still allows a reasonable number of
games per hour on your hardware.

### 10.3 Bigger Network

If you find the network training loss plateaus *and* increasing
simulations does not help, you may have hit network capacity. Adding
residual blocks or filters increases capacity at a roughly cubic cost
(more layers × more channels × more compute per evaluation).

For the research survey, do **not** scale up the network — keep it
small and didactic. Capacity is a problem to solve only after the
small system clearly works.

### 10.4 Hyperparameter Tuning

In rough order of impact when stuck:

1. **`c_puct`** — try 1.0, 1.5, 2.5. Higher is more exploratory;
   lower is more exploitative.
2. **Dirichlet `(α, ε)`** — increase `ε` to 0.4 if games are too
   draw-heavy.
3. **Temperature crossover threshold** — moves 15 vs 30 vs 50 affect
   game diversity vs decisiveness.
4. **Learning rate** — halve it if loss is unstable; double it if
   loss decreases too slowly.
5. **Replay buffer size** — smaller buffer = more responsive to
   recent data, more risk of overfitting recent self-play; larger =
   smoother but more staleness risk.

### 10.5 Architecture Changes (Last Resort)

- More residual blocks (depth)
- Wider blocks (more filters per block)
- Different activation (Mish, Swish — KataGo uses these)
- Auxiliary heads (KataGo predicts ownership, score, opponent reply
  policy — these all add useful gradient signal)

Any of these requires non-trivial code changes and should not be
attempted before the simpler levers are exhausted.

---

## 11. Comparison with NNUE Training

A direct side-by-side, building on [the architectural comparison in
the AlphaZero reference doc](AlphaZero_Reference.md#10-comparison-with-nnue):

| Aspect                | NNUE (HalfKAv2)                                | AlphaZero-style                              |
|-----------------------|------------------------------------------------|----------------------------------------------|
| Data origin           | External engine search → SFEN files (off-line) | Self-play with current network (on-line)     |
| Data is fixed?        | Yes — generated once, then reused              | No — replay buffer constantly turns over     |
| Bootstrap             | Not needed; teacher is external                 | Required; MCTS is the teacher                |
| Per-position targets  | (engine_score, game_result)                    | (mcts_visit_distribution, game_outcome)      |
| Score-side loss       | sigmoid MSE (probability space)                 | direct MSE (already in [-1, +1])              |
| Policy loss           | None                                            | cross-entropy vs MCTS distribution           |
| Lambda weights        | λ_score=1.0, λ_result=0.5                       | typically equal weights, sometimes value=1.5  |
| Score scaling         | divide centipawns by 600                        | not needed                                    |
| Optimizer             | Adam, lr=1e-3, weight_decay=1e-6                | SGD-momentum or Adam, lr ~1e-3, wd=1e-4     |
| Batch size            | 4,096–16,384 (small per-sample compute)         | 256–4,096 (each sample is a heavy conv pass) |
| Iterations needed     | ~200 epochs of fixed dataset                    | ~thousands of iterations of (self-play+train)|
| Validation set        | Held-out SFEN positions                         | Evaluation match against previous best       |
| Best-model tracking   | Lowest validation loss                          | Highest evaluation match win rate            |
| Early stopping        | Validation-loss-based                           | "Promotion fails repeatedly" → halt          |
| Per-eval cost (CPU)   | ~100 ns                                         | ~10–100 ms                                    |
| Total training compute| ~1 GPU-day for a starter model                  | ~30 GPU-days for a starter model              |

The most striking practical difference: **NNUE training is
deterministic given the dataset**, while AlphaZero training is
stochastic because the dataset is co-evolving. Two AlphaZero training
runs starting from the same seed on the same hardware will diverge
after a few generations, simply because slightly different self-play
games produce slightly different replay buffers.

---

## 12. Realistic Expectations for a Research Survey

The point of this survey is **understanding**, not strength. Concrete
expectations for the planned reference implementation:

- **Compute budget**: A single laptop or single-GPU machine, single
  process, no distributed self-play. Total training time per
  experiment: hours, not days.
- **Network**: Tiny by AlphaZero standards (4 residual blocks × 64
  filters, ~1M params). Small enough to evaluate on CPU in a few
  milliseconds per position.
- **MCTS budget**: 50–200 simulations per move. Enough to demonstrate
  the algorithm; nowhere near competitive strength.
- **Self-play volume**: ~10K–100K games per training run.
- **Strength target**: Stronger than random play. Stronger than the
  PST material evaluator on a small subset of positions would be a
  pleasant bonus, but not a hard goal.
- **Comparison target**: Where the NNUE pipeline produces a model that
  beats PST decisively (95%+ win rate), the AlphaZero reference
  implementation will *not*. That is expected. The interesting
  comparison is *qualitative*: what kinds of moves does each system
  prefer? Where do their evaluations agree or disagree? What is
  faster to train per unit of strength?

If at any point during the implementation work the question becomes
"why is this so slow / weak?", the answer is almost always: because
AlphaZero needs orders of magnitude more compute than NNUE to reach
the same strength. That is the survey's point — to make this
quantitatively obvious through hands-on experience.

---

## See Also

- [`doc/alphazero/AlphaZero_Reference.md`](AlphaZero_Reference.md) —
  architecture, MCTS, loss derivation, glossary, and reading list.
- [`doc/nnue/Model_Training_Reference.md`](../nnue/Model_Training_Reference.md)
  — the parallel training reference for the NNUE pipeline. Many
  hyperparameter discussions (early stopping, gradient clipping,
  evaluation matches) carry over.
- *AlphaZero pseudocode appendix*, arXiv:1712.01815 — the most
  precise spec of the original training loop.
- *Leela Chess Zero* and *KataGo* source code — modern engineering
  refinements on the original algorithm.
