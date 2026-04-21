# Bonanza Method Reference Guide

A technical reference for the Bonanza method of supervised evaluation
learning from professional Shogi game records (kifu). This document
covers the historical context, the learning formulation, and how the
approach fits into ShogiDojo's three-paradigm survey.

See also:

- [`doc/nnue/NNUE_Reference.md`](../nnue/NNUE_Reference.md) — the NNUE
  paradigm (learn from engine-generated scores).
- [`doc/alphazero/AlphaZero_Reference.md`](../alphazero/AlphaZero_Reference.md) —
  the AlphaZero paradigm (learn from self-play).

---

## Table of Contents

1. [What is the Bonanza Method?](#1-what-is-the-bonanza-method)
2. [Historical Context](#2-historical-context)
3. [The Learning Formulation](#3-the-learning-formulation)
4. [ShogiDojo's Implementation](#4-shogidojos-implementation)
5. [Data: Professional Game Records](#5-data-professional-game-records)
6. [Three-Paradigm Comparison](#6-three-paradigm-comparison)
7. [Strengths and Limitations](#7-strengths-and-limitations)
8. [Glossary](#8-glossary)
9. [References](#9-references)

---

## 1. What is the Bonanza Method?

The Bonanza method is a technique for training a Shogi evaluation
function (or, in our case, a move-prediction policy) by **supervised
learning on professional game records**. The core idea is simple:

> A professional player's move choices encode expert knowledge about
> which positions are good and which moves are strong. If we train a
> neural network to predict the moves professionals play, the network
> implicitly learns to evaluate positions — because predicting the
> right move requires understanding which positions the move leads to.

In ShogiDojo's implementation, we frame this as a classification
problem:

```
Input:   A board position (encoded as a tensor, same as AlphaZero)
Label:   The move the professional played (an index into the policy space)
Loss:    Cross-entropy between the network's predicted move distribution
         and the one-hot target (the professional's chosen move)
```

No search is performed during training. The network learns purely from
static position → move supervision. This makes the training loop much
simpler than either NNUE (which requires an engine to generate training
data) or AlphaZero (which requires MCTS self-play).

---

## 2. Historical Context

### Bonanza (2006)

Bonanza, created by Kunihito Hoki, revolutionized computer Shogi at
WCSC26 (the 26th World Computer Shogi Championship) in 2006. Before
Bonanza, evaluation functions were hand-tuned by programmers — a
laborious process that required deep Shogi expertise.

Bonanza's key innovation: **automatically tune millions of evaluation
parameters** by minimizing prediction error on a database of
professional games. Specifically, Bonanza used logistic regression on
game outcomes: given a position from a professional game, the
evaluation should predict that the side who eventually won has a
higher score.

### The "Bonanza Method" Legacy

The term "Bonanza method" (ボナンザメソッド) became shorthand in the
Japanese computer Shogi community for "learn evaluation weights from
professional game records." Many subsequent engines (Apery, Ponanza,
GPS Shogi) adopted variants of this approach.

The method's dominance lasted roughly a decade (2006–2018), until
NNUE (2018) and AlphaZero (2017) demonstrated that **self-generated
data** (from engine search or network self-play) could surpass
human-game-based training given sufficient compute.

### Modern Relevance

Today the Bonanza method is not competitive at the top level, but it
remains valuable for three reasons:

1. **Simplicity.** No search engine needed for data generation. No
   self-play infrastructure. Just a dataset and a standard training
   loop.
2. **Warm-starting.** A network pre-trained on professional games
   provides a much better starting point for AlphaZero self-play
   than random initialization. This accelerates early training
   significantly.
3. **Pedagogy.** Supervised learning from labeled data is the most
   familiar ML paradigm. Starting here before moving to NNUE or
   AlphaZero gives the team a smooth on-ramp.

---

## 3. The Learning Formulation

### Original Bonanza: Evaluation Regression

Bonanza's original formulation trains an *evaluation function* (not
a policy). Given a position from a game that Black eventually won:

```
Loss = max(0, margin - eval(position))
```

This pushes the evaluation to be positive (favoring Black) in
positions from games Black won, and negative in positions from games
Black lost. The margin prevents overfitting to the exact outcome.

### ShogiDojo: Policy Classification

ShogiDojo uses a modernized formulation: instead of learning an
evaluation, we learn a **move-prediction policy**. This directly
produces the probability distribution over legal moves that
AlphaZero's policy head computes — making the Bonanza-trained
network interchangeable with an AlphaZero-trained one.

```
Input:    position tensor x  (shape: 45 × 9 × 9)
Target:   index of the professional's move in the 9×9×139 action space
Loss:     cross-entropy(softmax(policy_logits), target_index)
```

The value head is trained simultaneously on the game outcome:

```
Value target:  +1 if STM won the game, −1 if lost, 0 if draw
Value loss:    MSE(tanh(value_output), value_target)
```

This dual-head training means the resulting network is architecturally
identical to an AlphaZero network and can be dropped into the
AlphaZero self-play loop as a warm-start.

### Why Classification Instead of Regression?

The original Bonanza used regression on game outcomes because its
evaluation function was a scalar. Our network has a policy head
(multi-class output), so classification is the natural formulation.
Additionally, modern deep learning tooling (cross-entropy loss,
softmax, GPU-accelerated batching) is optimized for classification,
making implementation straightforward.

---

## 4. ShogiDojo's Implementation

### Architecture

We reuse the AlphaZero network (`AlphaZeroNetwork` from
`src/shogi_dojo/alphazero/network.py`) without modification. Both
heads are trained:

- **Policy head** → cross-entropy against the professional's move
- **Value head** → MSE against the game outcome

This produces a network that is architecturally identical to an
AlphaZero-trained one, enabling warm-start experiments.

### Data Pipeline

```
CSA game files (professional kifu)
    ↓  python-shogi CSA parser
List of games: (starting SFEN, [USI moves], winner)
    ↓  replay on shogi.Board
Per-position samples: (board_state, pro_move, legal_moves, game_outcome)
    ↓  encode_position + encode_move
Training tensors: (input_tensor, target_index, legal_mask, value_target)
    ↓  PyTorch DataLoader
Batched training
```

### Training Loop

Standard epoch-based supervised learning:

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        policy_logits, value = network(batch["input"])
        policy_loss = cross_entropy(policy_logits, batch["target"])
        value_loss = MSE(value, batch["value_target"])
        loss = policy_loss + lambda_value * value_loss
        optimizer.step(loss)
```

No MCTS. No self-play. No replay buffer. The simplicity is the point.

### Dataset Modes: Map-Style vs Streaming

``src/shogi_dojo/bonanza/dataset.py`` provides two PyTorch-compatible
dataset classes. The trainer (:class:`BonanzaTrainer`) auto-detects
which one it was given and configures the ``DataLoader`` accordingly;
the user chooses which to use based on dataset size.

- **``SupervisedDataset``** (map-style). Loads every training sample
  into memory at construction. Simpler, permits standard
  ``shuffle=True`` in the DataLoader for globally-random sampling.
  Use this when the full dataset fits in RAM — roughly under 100K
  training samples (≈ 1,000 games).

- **``StreamingSupervisedDataset``** (``IterableDataset``). Produces
  samples on-the-fly by replaying games during iteration. Memory use
  is bounded by a sliding shuffle buffer (default 10,000 samples),
  regardless of how many CSA files are in the source list. Use this
  for large datasets — e.g. the full shogi-server wdoor archive
  (~167K CSA files, ~13M samples, ~200 GB if loaded eagerly).

  The sliding shuffle buffer fills to N samples, shuffles, yields
  N/2, and carries the remaining N/2 forward to mix with new
  arrivals. At ~80 samples per game this mixes positions from ~120
  different games per yield — enough to break intra-game correlation
  that would otherwise hurt SGD convergence.

  When used with ``DataLoader(num_workers=M)`` each worker process
  iterates a round-robin-assigned slice of the file list with its
  own seeded RNG, so no sample is produced twice per epoch.

Choose at the CLI:

```bash
# Map-style (default) — small dataset:
python train_bonanza.py --data small_games.csa

# Streaming — large dataset:
find data/wdoor -name '*.csa' > data/wdoor_files.txt
python train_bonanza.py --data-list data/wdoor_files.txt --streaming
```

### GPU Utilization Tuning

The default `configs/bonanza/default.yaml` includes three settings that
make a big difference for GPU-backed training; if you see low GPU
utilization in `nvidia-smi`, these are the first things to check:

- **`num_workers`** — DataLoader worker processes that parse CSA and
  encode tensors in parallel. With `num_workers=0` (the PyTorch default)
  all CSA parsing and sample encoding happens on the main thread, so
  the GPU sits idle between batches. The default config uses `8`; go
  as high as your CPU core count allows.
- **`pin_memory`** — when `true`, DataLoader staging buffers live in
  page-locked memory so the CPU→GPU transfer can happen asynchronously.
  Combined with `non_blocking=True` on the trainer's `.to(device)`
  calls (always on inside the trainer when using CUDA), transfers
  overlap with compute. CUDA-only; ignored on CPU.
- **`batch_size`** — the default AlphaZero-family network is tiny
  (~336K params). On a modern GPU, batch size must be large to keep
  the compute pipeline full. The default config uses `1024`; on an
  A100 or H100 you can probably push to `2048` or beyond.

The trainer logs **samples/second** at the end of every epoch. Use
that number to compare before/after changes: a CPU-bound pipeline
plateaus at a few thousand samples/sec even on a fast GPU, while a
well-tuned one on a modern GPU reaches tens of thousands.

### Checkpoint Cadence

Two independent checkpoint triggers:

- **`save_every`** (epochs) — save after every N epochs and always at
  the final epoch. Default `10`. Fine for small datasets.

- **`save_every_steps`** (optimizer steps) — save after every N
  optimizer steps, mid-epoch. Essential for long-running streaming
  training where a single epoch can take hours: without it, a
  mid-epoch crash loses all progress. Default in the shipped
  `default.yaml` is `5000` (≈ a few minutes at GPU speed). Set to
  `null` (or omit) to disable.

Step-based checkpoints are named ``bonanza_step_NNNNNNNN.pt``; epoch-
based are ``bonanza_epoch_NNNN.pt``. Both payloads include ``epoch``,
``global_step``, ``model_state_dict``, and ``optimizer_state_dict`` —
everything the trainer needs to resume.

### Resuming Training

Pass ``--resume PATH`` to ``train_bonanza.py`` to continue from a
saved checkpoint:

```bash
python train_bonanza.py --resume checkpoints/bonanza_epoch_0010.pt \
    --data-list data/wdoor_files.txt --streaming
```

Internally, ``BonanzaTrainer.resume_from(path)`` restores the model
and optimizer state and sets ``start_epoch = saved_epoch + 1`` and
``global_step = saved_global_step``. The next ``train()`` call picks
up from there, and step-numbered checkpoint filenames continue
counting from where the previous run left off.

Resuming from a **step-based** checkpoint skips the remainder of the
in-progress epoch — the trainer restarts cleanly at the next epoch.
Exact mid-epoch resume is not supported because the DataLoader/RNG
state is not persisted; for streaming datasets this is unavoidable
without a substantial bookkeeping overhead.

If ``training.epochs`` in the config is less than or equal to the
saved epoch (e.g. you resumed from the final checkpoint without
bumping the target), the trainer logs a warning and exits without
running any further updates.

Troubleshooting checklist if utilization is still low:

1. `nvidia-smi` during training — is the GPU at 100%? If yes, it is
   not the bottleneck; tune elsewhere (e.g. increase network size).
2. Run with `num_workers=0` briefly to confirm parallelism helps;
   it should slow training significantly.
3. Verify `pin_memory=True` is logged at trainer start (the trainer
   forces it off if the device is CPU).
4. Try `batch_size * 2` until OOM; each doubling raises GPU
   utilization until compute saturates.
5. For very large datasets, watch for DataLoader shared-memory
   limits: if workers crash with "bus error" or "Connection reset",
   raise the system's shared-memory quota (Docker/k8s `/dev/shm`
   default is 64 MB — bump to at least 2 GB).

---

## 5. Data: Professional Game Records

### CSA Format

CSA (Computer Shogi Association) is the standard machine-readable
format for Shogi game records. A CSA file contains:

```
V2.2
N+Sente Player Name
N-Gote Player Name
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA *
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
...
+
+7776FU
-3334FU
+2726FU
%TORYO
```

Each move is a line like `+7776FU` (Black moves a Pawn from 77 to 76).
The game ends with a result line (`%TORYO` = resignation,
`%SENNICHITE` = repetition draw, etc.).

python-shogi's `shogi.CSA.Parser.parse_str()` handles all parsing
and returns structured game data.

### Data Sources

Professional Shogi game records are available from various archives.
Common sources include:

- **shogi-server (Floodgate) kifu archives** — the operator of the
  Floodgate 24/7 Shogi server periodically publishes compressed CSA
  game archives via GitHub Releases at
  [github.com/shogi-server/shogi-server/releases](https://github.com/shogi-server/shogi-server/releases).
  The
  [2026-04-10 release](https://github.com/shogi-server/shogi-server/releases/tag/2026-04-10)
  is what the team currently uses as a starting dataset.

  The release ships two 7-Zip archives covering successive years,
  each containing one CSA file per game:

  | Archive          | Games   | Notes                                                |
  |------------------|---------|------------------------------------------------------|
  | `wdoor2025.7z`   | 128,843 | Full 2025 year of Floodgate games.                    |
  | `wdoor2026.7z`   |  37,897 | 2026 year-to-date at the time of the 2026-04-10 cut. |
  | **Combined**     | **166,740** | ~13M training positions after replay (~80/game avg). |

  Extract with ``7z x <archive>.7z`` (install ``p7zip-full`` on
  Debian/Ubuntu). Most games are engine-vs-engine; many involve
  strong human players as well. Good for bootstrapping — filter by
  player rating if targeting pro-level play.

  Minimal workflow once extracted:

  ```bash
  # Extract. Each archive creates a year subdirectory (wdoor/2025/,
  # wdoor/2026/); CSA files live one level down, not flat under
  # data/wdoor/.
  mkdir -p data/wdoor
  (cd data/wdoor && 7z x ../../wdoor2025.7z && 7z x ../../wdoor2026.7z)

  # Build a file list. With ~167K CSA files, passing them all on the
  # command line would exceed the argv limit (~2 MB on Linux; the full
  # dataset is ~15 MB of paths). Use --data-list to read paths from a
  # file instead.
  find data/wdoor -name '*.csa' > data/wdoor_files.txt
  wc -l data/wdoor_files.txt   # should report 166740

  python train_bonanza.py --config configs/bonanza/tiny.yaml \
      --data-list data/wdoor_files.txt --min-ply 10
  ```

  For a quick smoke test on a tiny subset, ``--data`` with a
  targeted glob works because there are only ~100 files to pass:

  ```bash
  # Just one day's worth of games (argv-safe):
  python train_bonanza.py --config configs/bonanza/tiny.yaml \
      --data data/wdoor/2025/*20250215*.csa --min-ply 10
  ```
- **Japan Shogi Association records** — official kifu from
  professional title matches. Available through various fan sites
  and databases.
- **WCSC archives** — games from the World Computer Shogi
  Championship, useful for training on strong computer play.

For the survey, even a small dataset (1,000–10,000 games) is
sufficient to demonstrate the method. The network will learn basic
piece-value awareness and common opening moves.

### Data Quality Considerations

- **Filter by player strength.** Professional games teach better
  than amateur games. If mixing sources, weight professional games
  higher or filter by rating.
- **Exclude early resignations.** Games that end before move 30 may
  have been disconnections or errors, not meaningful play.
- **Balance by opening.** If the dataset is dominated by one opening
  (e.g., static rook), the network will overfit to that opening
  style.

---

## 6. Three-Paradigm Comparison

| Aspect               | Bonanza (supervised)    | NNUE                        | AlphaZero                    |
|-----------------------|-------------------------|-----------------------------|------------------------------|
| Data source           | Professional games      | Engine self-play (gensfen)  | Network self-play (MCTS)     |
| Human data required?  | **Yes** — the key input | No                          | No                           |
| Search at training?   | No                      | Yes (alpha-beta in gensfen) | Yes (MCTS in self-play)      |
| Search at play time?  | Optional (move ordering)| Yes (alpha-beta)            | Yes (MCTS)                   |
| Training complexity   | **Simplest**            | Medium                      | Complex                      |
| Strength ceiling      | Limited by human data   | Limited by search depth     | Limited by compute           |
| Warm-start value      | **High** — for AlphaZero| N/A (different architecture)| N/A (is the target)          |
| Implementation lines  | ~400                    | ~900                        | ~1,800                       |

### The Key Trade-Off

Bonanza is the **fastest to get running** but has the **lowest
strength ceiling**: the network can never exceed the quality of the
training data (professional games). NNUE and AlphaZero generate
their own data at arbitrary quality, so they can in principle
surpass any fixed dataset — but they need orders of magnitude more
compute to do so.

For a team with limited compute but access to professional game
databases, Bonanza is the pragmatic starting point. The trained
policy can then be used to warm-start AlphaZero self-play (Phase
3.5 in the roadmap), combining the advantages of both approaches.

---

## 7. Strengths and Limitations

### Strengths

- **No infrastructure.** No self-play engine, no MCTS, no replay
  buffer. Just a dataset, a network, and a training loop.
- **Fast iteration.** Training on a pre-existing dataset is much
  faster than generating new data. Hyperparameter sweeps that
  would take days with AlphaZero take minutes with Bonanza.
- **Human-interpretable targets.** "Predict the professional's
  move" is immediately understandable; "match the MCTS visit
  distribution" requires algorithmic context.
- **Warm-start capability.** The trained weights transfer directly
  to AlphaZero's architecture.

### Limitations

- **Data dependency.** Requires professional game records. These
  are available for Shogi but not for every game variant.
- **Strength ceiling.** The network cannot learn patterns that
  professionals don't demonstrate. Unusual or theoretically
  superior lines that no professional has played are invisible.
- **Distribution mismatch.** The positions in professional games
  are not the same positions the network will encounter during
  self-play or search. Professionals rarely play into losing
  positions, so the network gets little training signal on "how
  to play when losing." This is the fundamental limitation that
  self-play-based methods overcome.
- **No search integration.** The training loop does not use search,
  so the network does not learn to work *with* a search algorithm.
  AlphaZero's policy is specifically shaped to be useful inside
  MCTS; Bonanza's policy is shaped to match human judgment.

---

## 8. Glossary

| Term                  | Definition |
|-----------------------|------------|
| **Bonanza method**    | Supervised training of evaluation/policy weights from professional game records. Named after the Bonanza engine (2006). |
| **CSA format**        | Computer Shogi Association standard format for game records. Machine-readable text with one move per line. |
| **Kifu (棋譜)**       | A Shogi game record. Can be in CSA, KIF, or other formats. |
| **Policy prediction** | Training a network to predict which move a strong player would choose, framed as multi-class classification. |
| **Warm-start**        | Initializing a network's weights from a pre-trained checkpoint (e.g., Bonanza-trained) before starting a different training process (e.g., AlphaZero self-play). |

---

## 9. References

- K. Hoki, *"Optimal control of minimax search results to learn
  positional evaluation,"* IPSJ Journal, 2006. The original Bonanza
  paper (in Japanese).
- K. Hoki and T. Kaneko, *"Large-scale optimization for evaluation
  functions with minimax search,"* JAIR 49, 2014. The English-language
  reference for the Bonanza method's mathematical formulation.
- T. Kaneko, *"Computer Shogi — survey,"* IPSJ Magazine, 2012. A
  survey of computer Shogi methods including the Bonanza approach.
  (In Japanese, widely referenced in English discussions.)

### Cross-References Within ShogiDojo

- [`AlphaZero_Reference.md`](../alphazero/AlphaZero_Reference.md)
- [`AlphaZero_Training_Reference.md`](../alphazero/AlphaZero_Training_Reference.md)
- [`NNUE_Reference.md`](../nnue/NNUE_Reference.md)
- [`Model_Training_Reference.md`](../nnue/Model_Training_Reference.md)
- [`Deployment.md`](Deployment.md) — single-source-of-truth contract for porting the trained network to a non-Python runtime (encoder layout, policy index decoding, value perspective convention).
