# AlphaZero vs NNUE — Operational Comparison

This document expands the side-by-side table in
[`AlphaZero_Reference.md` §10](AlphaZero_Reference.md#10-comparison-with-nnue)
into a standalone comparison with concrete numbers drawn from both
ShogiDojo's implementations and the published literature.

It is intended as a quick-reference sheet for team discussions,
*not* a research paper — opinions are labeled as such, and all
numbers are order-of-magnitude unless otherwise cited.

---

## 1. Architecture at a Glance

|                        | NNUE (HalfKAv2)                            | AlphaZero-style                                  |
|------------------------|--------------------------------------------|--------------------------------------------------|
| Network type           | Sparse embedding → 3-layer MLP             | Deep residual CNN (two-headed)                   |
| Input representation   | ~40 sparse indices out of 19,638           | Dense 45–119 planes of 9×9 floats               |
| Output                 | Single scalar (centipawn score)             | Policy logits (139×9×9) + scalar value (tanh)    |
| Parameters (default)   | 5,045,089 (~5 M)                           | 336,398 (~340 K survey default; paper ~46 M)     |
| Where params live      | 99.7% in Feature Transformer (lookup-only) | Distributed across all conv + FC layers          |
| Activation             | ClippedReLU [0, 1]                         | ReLU (+ BatchNorm in trunk)                      |

## 2. Inference Cost

|                          | NNUE                       | AlphaZero                              |
|--------------------------|----------------------------|----------------------------------------|
| Per-eval (CPU)           | ~100 ns (incremental)      | ~5–50 ms (full conv forward pass)      |
| Per-eval (GPU)           | Not typical; CPU is faster | ~0.5–5 ms batched                      |
| Evals per move (search)  | 10⁶ – 10⁷ (alpha-beta)    | 10² – 10³ (MCTS simulations)          |
| Effective move time      | < 1 ms feasible            | ≥ 100 ms typical                       |

**Key insight.** NNUE achieves fast total move time via *many cheap
evaluations* (alpha-beta visits millions of nodes but each eval costs
nanoseconds thanks to incremental accumulator updates). AlphaZero
achieves good total move time via *few expensive evaluations* (MCTS
visits hundreds of nodes, each costing milliseconds, but each visit
carries much more information — a policy prior and a value estimate).

Neither strategy dominates the other. At the same wall-clock budget,
NNUE + alpha-beta sees more of the game tree (breadth+depth) while
AlphaZero sees less but with a learned search policy that spends
visits more selectively.

## 3. Training Data

|                            | NNUE                                      | AlphaZero                                   |
|----------------------------|-------------------------------------------|----------------------------------------------|
| Data source                | External engine search → SFEN files       | Self-play with current network               |
| Data is pre-generated?     | Yes — generated once, train many epochs    | No — continuously produced and aged out      |
| Per-position targets       | (engine_score, game_result)               | (MCTS_visit_distribution, game_outcome)      |
| Human data required?       | No (engine self-play)                     | No (network self-play)                       |
| Data volume for starter    | ~1 M positions (one SFEN file)             | ~10 K games (generated live)                 |
| Data volume for strength   | ~100 M – 1 B positions                    | ~1 M – 10 M games over many iterations       |
| Storage                    | 5–100 GB on disk (text SFEN)               | In-memory replay buffer (~100 MB)            |

**Trade-off.** NNUE's off-line data generation is simpler to manage
(generate once, train many times, no moving parts). AlphaZero's
on-line data generation is more compute-intensive but avoids the
bootstrapping chicken-and-egg problem: you do not need an existing
strong engine to generate labels. For a team with no strong engine
and no Shogi game databases, AlphaZero's self-play approach is
conceptually appealing. For a team with a strong engine already
(e.g., YaneuraOu or the gensfen tool), NNUE's off-line approach is
far more compute-efficient.

## 4. Training Compute

|                            | NNUE                               | AlphaZero (survey default)           |
|----------------------------|-------------------------------------|--------------------------------------|
| GPU requirement            | Optional; trains fine on CPU        | Strongly recommended                 |
| Time to "beats random"     | ~10 minutes                         | ~1 hour                              |
| Time to "beats PST"        | ~1 GPU-hour                         | ~1 GPU-day                           |
| Time to competitive        | ~1–10 GPU-days                      | ~100s of GPU-days                    |
| Bottleneck                 | Data generation (gensfen depth)     | Self-play (MCTS evals per game)      |
| Epochs / iterations        | ~200 epochs of a fixed dataset      | ~50–500 iterations of (play+train)   |
| Checkpoint size            | ~19 MB                              | ~1–2 MB (survey net is smaller)      |

## 5. Deployment

|                            | NNUE                               | AlphaZero                            |
|----------------------------|-------------------------------------|--------------------------------------|
| Engine integration         | Designed for it (int8/int16 quant)  | Requires GPU or large-batch CPU      |
| Mobile-friendly?           | Yes — runs on ARM/NEON at ~1 MHz   | No without heavy optimization        |
| Binary size                | ~20 MB for weights                  | ~1–50 MB depending on network size   |
| Inference latency floor    | ~1 μs (incremental update)          | ~1 ms (GPU) / ~50 ms (CPU)          |

## 6. Development Complexity

|                            | NNUE                               | AlphaZero                            |
|----------------------------|-------------------------------------|--------------------------------------|
| Lines of Python (ShogiDojo)| ~900 (src/shogi_dojo/nnue/)         | ~1,800 (src/shogi_dojo/alphazero/)   |
| External rule dependency   | None (SFEN parser only)             | python-shogi for legal moves         |
| Search implementation      | External (engine provides α-β)      | Built-in MCTS (~400 lines)           |
| Key subtle invariant       | -1 padding convention (dataset ↔ model) | Perspective sign-flip on backup   |
| Hardest-to-debug failure   | Quantization scale mismatch         | Value collapse / policy collapse     |

## 7. When to Use Which

| Scenario                                        | Recommended       |
|-------------------------------------------------|-------------------|
| Building a Shogi app that runs on phones         | NNUE              |
| Exploring a new game variant with no existing engine | AlphaZero     |
| Maximizing playing strength per GPU-hour         | NNUE              |
| Learning how neural-network-guided search works  | AlphaZero         |
| Need sub-millisecond move times                  | NNUE              |
| Want to avoid any game-specific feature engineering | AlphaZero      |
| Team has a strong engine for generating labels   | NNUE              |
| Research on search + learning interactions       | AlphaZero         |

**In modern strongest engines, the answer is increasingly "both."**
Stockfish uses NNUE evaluation inside an alpha-beta framework that
was *informed* by AlphaZero-era insights (better time management,
less hand-tuned search, neural evaluation). dlshogi and AobaZero use
AlphaZero-style training to produce a policy/value network that is
then queried inside MCTS. Studying both paradigms — as ShogiDojo
now does — is preparation for understanding and contributing to
these hybrid systems.

---

## See Also

- [`AlphaZero_Reference.md`](AlphaZero_Reference.md) — architecture,
  MCTS, and glossary.
- [`AlphaZero_Training_Reference.md`](AlphaZero_Training_Reference.md)
  — training loop, hyperparameters, and failure modes.
- [`../nnue/NNUE_Reference.md`](../nnue/NNUE_Reference.md) — the NNUE
  side of the comparison.
- [`../nnue/Model_Training_Reference.md`](../nnue/Model_Training_Reference.md)
  — NNUE training methodology.
