# AlphaZero Reading List

An annotated bibliography for the ShogiDojo team. Entries are grouped
by topic and ordered roughly from "read first" to "read when you need
it." All links checked as of April 2026.

---

## 1. The Core Papers

These three papers define the trajectory from human-assisted to
fully self-play–based game AI. Read them in order.

### AlphaGo (2016)

> D. Silver, A. Huang, C. J. Maddison, et al.
> *"Mastering the game of Go with deep neural networks and tree search."*
> Nature 529, 484–489 (2016).

The original. Defeated a top human Go player (Lee Sedol) for the
first time. Uses a policy network trained on human games, a value
network trained on self-play, and a fast rollout policy for Monte
Carlo simulations. Many of these components were later simplified
out in AlphaGo Zero.

**Read for:** the historical context and the idea that a learned
policy can guide MCTS.

### AlphaGo Zero (2017)

> D. Silver, J. Schrittwieser, K. Simonyan, et al.
> *"Mastering the game of Go without human knowledge."*
> Nature 550, 354–359 (2017).

Removes human data entirely. Introduces the single two-headed
network (policy + value), trains purely from self-play, and replaces
rollouts with the value head's estimate. The blueprint that AlphaZero
generalizes.

**Read for:** the core self-play bootstrap algorithm; the PUCT
formula; the loss function (value MSE + policy cross-entropy).

### AlphaZero (2017 / 2018)

> D. Silver, T. Hubert, J. Schrittwieser, et al.
> *"A general reinforcement learning algorithm that masters chess,
> shogi, and Go through self-play."*
> Science 362, 1140–1144 (2018).
> Earlier preprint: arXiv:1712.01815 (2017).

Generalizes AlphaGo Zero to chess and Shogi with the same code (only
the input/output encoding changes per game). The **appendix
pseudocode** in the arXiv version is the most precise public
specification of the algorithm.

**Read for:** Shogi-specific input planes (§ Methods, Table S3);
the pseudocode; the Shogi result (defeated Elmo after 9 hours on
5,000 TPUs).

---

## 2. Open-Source Implementations

Working codebases you can run, read, and diff against our own.

### Leela Chess Zero (LCZero)

- Website: [lczero.org](https://lczero.org/)
- Code: [github.com/LeelaChessZero](https://github.com/LeelaChessZero)

The most active open-source AlphaZero-style chess engine. Distributed
self-play across ~1,000 volunteer contributors. Excellent engineering
reference for:

- Distributed self-play worker architecture
- Network format and training pipeline (Python + C++)
- Dynamic `c_puct` (log-scaling variant)
- First Play Urgency (FPU) tuning

### KataGo

- Code: [github.com/lightvector/KataGo](https://github.com/lightvector/KataGo)
- Paper: D. Wu, *"Accelerating Self-Play Learning in Go,"* arXiv:1902.10565

A Go engine that refines AlphaZero with several clever ideas:

- **Auxiliary policy targets** (predicting the opponent's reply)
  that add useful gradient signal
- **Ownership prediction** (which stones are alive at the end)
- **Score-based value head** (predicting the margin, not just
  win/loss) — analogous to NNUE's score loss
- **Playout cap randomization** for more efficient self-play

**Read for:** practical improvements that apply to any game, and a
clear discussion of compute efficiency.

### dlshogi (Deep Learning Shogi)

- Code: [github.com/TadaoYamaoka/DeepLearningShogi](https://github.com/TadaoYamaoka/DeepLearningShogi)

The closest open-source reference for Shogi-specific AlphaZero
engineering. Written in C++ with Python training scripts. Useful for:

- Shogi-specific input planes and move encoding (cross-check our
  `encoding.py` against their implementation)
- MCTS with virtual loss for GPU batching
- Engine integration (USI protocol)

### AobaZero

- Code: [github.com/kobanium/aobazero](https://github.com/kobanium/aobazero)

A Japanese-led open-source AlphaZero project for Shogi. Smaller
community than LCZero but directly relevant to our domain.

**Read for:** Japanese-language discussion of Shogi-specific design
choices; another cross-reference for encoding conventions.

---

## 3. Related Paradigms

### MuZero (2019)

> J. Schrittwieser, I. Antonoglou, T. Hubert, et al.
> *"Mastering Atari, Go, chess, and shogi by planning with a learned
> model."*
> Nature 588, 604–609 (2020).

Removes the requirement that the search knows the game rules. The
network learns a *dynamics model* (state × action → next latent
state) alongside the policy and value heads. Out of scope for
ShogiDojo's current survey but important to know exists.

**Read for:** understanding where the field went after AlphaZero;
"model-based RL for games."

### NNUE (2018)

> Yu Nasu,
> *"Efficiently Updatable Neural-Network-based Evaluation Functions
> for Computer Shogi."*
> WCSC28 (2018). (In Japanese; widely discussed in English on the
> Stockfish / computer chess forums.)

The NNUE architecture that ShogiDojo's primary pipeline implements.
Originally published in Japanese for the World Computer Shogi
Championship.

**Read for:** the original motivation (speed of incremental updates)
and how it fits into a classical alpha-beta engine.

### Stockfish NNUE PyTorch Wiki

- URL: [official-stockfish.github.io/docs/nnue-pytorch-wiki](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)

The most comprehensive English-language reference on NNUE
architecture, quantization, and training. Written for chess but the
concepts transfer to Shogi with minimal adaptation. Referenced
throughout ShogiDojo's `doc/nnue/` documentation.

---

## 4. Tutorials and Pedagogical Material

### "A Simple Alpha(Go) Zero Tutorial"

A widely cited blog post and minimal Connect4 implementation that
captures the essential AlphaZero algorithm in ~300 lines of Python.
Multiple versions exist (search for "alpha zero tutorial connect4");
most follow the same structure.

**Read for:** seeing the algorithm with all game-specific complexity
stripped away. Good warm-up before reading the full paper.

### AlphaZero Pseudocode Appendix

The arXiv version of the AlphaZero paper (1712.01815) includes a
pseudocode appendix that is more precise than the prose. After
reading ShogiDojo's reference docs, reading the pseudocode directly
is highly recommended — it is short, unambiguous, and maps almost
line-for-line onto our `mcts.py` and `self_play.py`.

---

## 5. Hybrid and Future Directions

### Stockfish's AlphaZero-Inspired Evolution

Stockfish (chess) did *not* adopt AlphaZero's architecture, but
AlphaZero's dominance motivated Stockfish to:

1. Replace its hand-crafted evaluation with NNUE (2020).
2. Modernize its search heuristics (less hand-tuning, more data-
   driven parameter optimization via SPSA).
3. Adopt ideas from RL for tuning search parameters (Leela's
   influence on time management).

**Relevance:** a case study in how the NNUE paradigm absorbed
lessons from the AlphaZero paradigm without switching architectures.

### Distillation: AlphaZero Policy → NNUE

An active research direction: train an AlphaZero-style network via
self-play to produce high-quality move-selection data, then
*distill* that data back into an NNUE-style evaluator for deployment.
This combines AlphaZero's data-generation strength with NNUE's
inference speed.

**Relevance:** a natural next step if ShogiDojo's survey produces
an AlphaZero network whose policy is noticeably better than the
gensfen-based NNUE's implicit move preferences.

---

## Cross-References Within ShogiDojo

| Document | What it covers |
|---|---|
| [`AlphaZero_Reference.md`](AlphaZero_Reference.md) | Architecture, MCTS, encoding, glossary |
| [`AlphaZero_Training_Reference.md`](AlphaZero_Training_Reference.md) | Training loop, hyperparameters, failure modes |
| [`AlphaZero_vs_NNUE.md`](AlphaZero_vs_NNUE.md) | Side-by-side operational comparison |
| [`../nnue/NNUE_Reference.md`](../nnue/NNUE_Reference.md) | HalfKAv2 architecture reference |
| [`../nnue/Model_Training_Reference.md`](../nnue/Model_Training_Reference.md) | NNUE training methodology |
