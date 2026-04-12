# NNUE Reference Guide

A comprehensive technical reference for the NNUE (Efficiently Updatable Neural Network) architecture as implemented in ShogiDojo. This document is aimed at developers who are new to computer Shogi and want to understand how the pieces fit together.

## Table of Contents

1. [Shogi Basics for Programmers](#1-shogi-basics-for-programmers)
2. [NNUE Overview](#2-nnue-overview)
3. [HalfKAv2 Feature Encoding](#3-halfkav2-feature-encoding)
4. [Network Architecture](#4-network-architecture)
5. [Training Methodology](#5-training-methodology)
6. [SFEN Data Format](#6-sfen-data-format)
7. [Glossary](#7-glossary)

---

## 1. Shogi Basics for Programmers

### The Board

Shogi is played on a **9×9 board** (81 squares). By convention:

- **Ranks** (rows) are numbered 1–9 from the top (White's side) to the bottom (Black's side).
- **Files** (columns) are numbered 9–1 from right to left (from Black's perspective).

In ShogiDojo, squares are indexed 0–80, row by row from the top-left corner:

```
         File: 9   8   7   6   5   4   3   2   1
              ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
  Rank 1      │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │  ← White's back rank
              ├───┼───┼───┼───┼───┼───┼───┼───┼───┤
  Rank 2      │ 9 │10 │11 │12 │13 │14 │15 │16 │17 │
              ├───┼───┼───┼───┼───┼───┼───┼───┼───┤
  Rank 3      │18 │19 │20 │21 │22 │23 │24 │25 │26 │  ← White's promotion zone
              ├───┼───┼───┼───┼───┼───┼───┼───┼───┤
  Rank 4      │27 │28 │29 │30 │31 │32 │33 │34 │35 │
              ├───┼───┼───┼───┼───┼───┼───┼───┼───┤
  Rank 5      │36 │37 │38 │39 │40 │41 │42 │43 │44 │  ← Center
              ├───┼───┼───┼───┼───┼───┼───┼───┼───┤
  Rank 6      │45 │46 │47 │48 │49 │50 │51 │52 │53 │
              ├───┼───┼───┼───┼───┼───┼───┼───┼───┤
  Rank 7      │54 │55 │56 │57 │58 │59 │60 │61 │62 │  ← Black's promotion zone
              ├───┼───┼───┼───┼───┼───┼───┼───┼───┤
  Rank 8      │63 │64 │65 │66 │67 │68 │69 │70 │71 │
              ├───┼───┼───┼───┼───┼───┼───┼───┼───┤
  Rank 9      │72 │73 │74 │75 │76 │77 │78 │79 │80 │  ← Black's back rank
              └───┴───┴───┴───┴───┴───┴───┴───┴───┘
```

### The Pieces

Shogi has 8 base piece types. All except King and Gold can promote when entering the opponent's territory (the furthest 3 ranks).

| ID | Piece          | Japanese   | Promotes To    | Movement Summary                          |
|----|----------------|------------|----------------|--------------------------------------------|
| 0  | Pawn (P)       | 歩 fu      | Tokin (+P)     | 1 square forward                           |
| 1  | Lance (L)      | 香 kyosha  | ProLance (+L)  | Any number of squares forward              |
| 2  | Knight (N)     | 桂 keima   | ProKnight (+N) | 2 forward + 1 sideways (like chess knight but only forward) |
| 3  | Silver (S)     | 銀 gin     | ProSilver (+S) | 1 square diagonally, or 1 forward          |
| 4  | Gold (G)       | 金 kin     | —              | 1 square in any direction except diagonally backward |
| 5  | Bishop (B)     | 角 kaku    | Horse (+B)     | Any number of squares diagonally           |
| 6  | Rook (R)       | 飛 hisha   | Dragon (+R)    | Any number of squares orthogonally         |
| —  | King (K)       | 玉 ou      | —              | 1 square in any direction                  |

All promoted pieces except Horse and Dragon move like Gold. Horse = Bishop + King movement. Dragon = Rook + King movement.

### Captures and Drops (持ち駒)

The defining feature of Shogi compared to chess: **captured pieces switch sides and can be dropped back onto any empty square** on a future turn. Promoted pieces revert to their base form when captured.

This means:
- Pieces are never permanently removed from the game.
- The "hand" (持ち駒) is a pool of captured pieces available for dropping.
- Material advantage works differently — captured pieces become your own resources.

Maximum counts in hand (per player):

| Piece  | Max in Hand | Reason                                    |
|--------|-------------|-------------------------------------------|
| Pawn   | 18          | 9 per side × 2 sides (minus kings, etc.)  |
| Lance  | 4           | 2 per side × 2 sides                       |
| Knight | 4           | 2 per side × 2 sides                       |
| Silver | 4           | 2 per side × 2 sides                       |
| Gold   | 4           | 2 per side × 2 sides                       |
| Bishop | 2           | 1 per side × 2 sides                       |
| Rook   | 2           | 1 per side × 2 sides                       |

### Players

- **Black (Sente / 先手)**: Moves first. Uppercase letters in SFEN.
- **White (Gote / 後手)**: Moves second. Lowercase letters in SFEN.

> Note: Confusingly, in Shogi "Black" moves first (unlike chess where "White" moves first).

---

## 2. NNUE Overview

### What Problem Does NNUE Solve?

A Shogi engine needs to **evaluate positions** — assign a numerical score that indicates who is winning and by how much. During game-tree search, the engine evaluates millions of positions per second. The evaluation function must be:

1. **Accurate**: Capture subtle positional nuances beyond simple material counting.
2. **Fast**: Run in nanoseconds, not milliseconds.

Traditional hand-crafted evaluation functions are fast but miss complex patterns. Deep neural networks are accurate but far too slow for real-time search. NNUE bridges this gap.

### Key Innovation: Incremental Updates

In a search tree, consecutive positions differ by just one move (one piece changes location). NNUE exploits this:

1. The **Feature Transformer** (the big, expensive part) produces an "accumulator" — a dense vector summarizing the position.
2. When a piece moves, only a few input features change. The accumulator can be **incrementally updated** by subtracting the old feature's weight and adding the new one.
3. The **Evaluation Network** (the small, cheap part) takes the accumulator and produces a score in a few dozen operations.

Result: near-neural-network accuracy at near-handcrafted speed.

> ShogiDojo focuses on **training** the NNUE. The incremental update logic lives in the Shogi engine that uses the trained weights at inference time.

### Why "Half"?

The position is evaluated from **two perspectives** simultaneously:

- **STM (Side-To-Move)**: How does the position look to the player whose turn it is?
- **NSTM (Not-Side-To-Move)**: How does it look to the opponent?

Each perspective independently produces a feature vector. The two vectors are concatenated before feeding into the evaluation network. This lets the model learn asymmetric patterns — for example, having the initiative (it's your turn) is inherently different from waiting (it's the opponent's turn), even in the same board position.

---

## 3. HalfKAv2 Feature Encoding

### The Name Explained

**HalfKAv2** breaks down as:

| Part     | Meaning                                              |
|----------|------------------------------------------------------|
| **Half** | Two perspectives (STM + NSTM), each producing "half" the input |
| **K**    | King position used as a key to select feature weights |
| **A**    | All other pieces are encoded as features              |
| **v2**   | Second version, with improved king-position handling  |

### Feature Space

For each perspective, the input features encode:

#### Board Features (pieces on the board, excluding kings)

Each non-king piece produces one active feature based on:
- Its **piece type** (13 types: 7 base + 6 promoted)
- Its **color** (Black or White)
- Its **square** (0–80)

```
board_index = piece_type × 2 × 81 + color × 81 + square
```

Total board features: **13 × 2 × 81 = 2,106**

#### Hand Features (pieces in hand)

Hand pieces use **thermometer encoding**. Instead of encoding the count as a single number, we activate one feature per held piece:

```
Example: 3 Pawns in hand

Pawn slots:  [0] [1] [2] [3] [4] [5] ... [17]
Active?       ✓   ✓   ✓   ✗   ✗   ✗       ✗

→ Feature indices 0, 1, 2 are active.
```

Why thermometer encoding? It gives the network a natural "monotonic" signal — more pieces always means a superset of features are active. This is easier for the network to learn from than a single scalar count.

Slot ranges per piece type per side:

| Piece  | Slots | Cumulative Offset |
|--------|-------|-------------------|
| Pawn   | 0–17  | 0                 |
| Lance  | 0–3   | 18                |
| Knight | 0–3   | 22                |
| Silver | 0–3   | 26                |
| Gold   | 0–3   | 30                |
| Bishop | 0–1   | 34                |
| Rook   | 0–1   | 36                |
| **Total** | **38 per side** | |

Hand features per side: **38**. Both sides: **76**.

#### Combined Feature Dimension

```
Per perspective = Board + Hand = 2,106 + 76 = 2,182
```

### King Buckets

A Pawn on square 40 (center) means very different things depending on king safety:
- Is the king castled on the back rank?
- Is the king exposed in the center?
- Is the king advancing for an attack?

**King buckets** let the network learn separate piece-value patterns for different king positions. Instead of a single set of 2,182 features, we have one set **per bucket**.

ShogiDojo uses **9 king buckets** (one per rank):

```
King on Rank 1 (back rank)   → Bucket 0   — typical defensive position
King on Rank 2               → Bucket 1   — common castled position
King on Rank 3               → Bucket 2
...
King on Rank 9 (opponent's back rank) → Bucket 8  — very aggressive king
```

This means each perspective selects one bucket, and uses that bucket's feature offset:

```
full_index = king_bucket × 2,182 + feature_index
```

**Total input dimension**: 9 × 2,182 = **19,638**

Only one bucket is active per position, so despite the large total dimension, only ~40 features are active (the input is extremely sparse).

### Left-Right Symmetry (File Mirroring)

Shogi is horizontally symmetric — a position with the king on file 9 is strategically equivalent to the mirror image with the king on file 1. To exploit this, we mirror squares on the right half of the board to the left half:

```
Original files:  9  8  7  6  5  4  3  2  1
                 0  1  2  3  4  3  2  1  0   ← After mirroring (0-indexed)
                 ─────────────┤
                    Left half  │  Right half (mirrored to left)
                               │
                          Center file
```

This halves the effective number of distinct file positions the network must learn.

### Two-Perspective Computation

For every position, features are computed from **both** perspectives:

**Black's perspective** (no transformation):
- Board seen as-is
- Apply file mirroring
- King bucket from Black's king position

**White's perspective** (flipped view):
- Board rotated 180° (square 0 ↔ square 80)
- All piece colors swapped (Black ↔ White)
- Apply file mirroring
- King bucket from White's (flipped) king position

This ensures both perspectives see the board "from their own side" — as if they're always sitting at the bottom of the board with their own pieces nearby.

### Putting It All Together

For a single position, the feature extraction pipeline produces:

```
Position
  ├─ Parse SFEN → pieces, kings, hand, side-to-move
  ├─ Black's perspective:
  │    ├─ King bucket = rank of Black's king (after mirror)
  │    ├─ For each piece: compute index using bucket + type + color + mirrored square
  │    └─ For each hand piece: compute index using bucket + thermometer slot
  │    → ~40 active indices out of 19,638
  │
  └─ White's perspective:
       ├─ Flip board 180°, swap colors
       ├─ King bucket = rank of White's king (in flipped coordinates)
       ├─ For each piece: same index formula with flipped square and swapped color
       └─ For each hand piece: same formula with swapped color
       → ~40 active indices out of 19,638

Output: (stm_indices, nstm_indices)
  stm_indices  = perspective of the player whose turn it is
  nstm_indices = perspective of the opponent
```

---

## 4. Network Architecture

### Architecture Diagram

```
                       ┌──────────────┐
                       │   Position   │
                       │   (SFEN)     │
                       └──────┬───────┘
                              │
                    ┌─────────┴─────────┐
                    │ Feature Extraction │
                    │    (features.py)   │
                    └─────────┬─────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
     STM indices (~40)              NSTM indices (~40)
              │                               │
              ▼                               ▼
    ┌───────────────────┐           ┌───────────────────┐
    │     Sparse        │           │     Sparse        │
    │  Weight Lookup    │           │  Weight Lookup    │
    │   & Sum           │           │   & Sum           │
    │  (shared weights) │           │  (shared weights) │
    │  19,638 → 256     │           │  19,638 → 256     │
    └────────┬──────────┘           └────────┬──────────┘
             │                               │
             └───────────┬───────────────────┘
                         │ Concatenate
                         ▼
                   ┌───────────┐
                   │  512-dim  │
                   └─────┬─────┘
                         │ ClippedReLU [0, 1]
                         ▼
                   ┌───────────┐
                   │ Linear    │
                   │ 512 → 32  │
                   └─────┬─────┘
                         │ ClippedReLU [0, 1]
                         ▼
                   ┌───────────┐
                   │ Linear    │
                   │  32 → 32  │
                   └─────┬─────┘
                         │ ClippedReLU [0, 1]
                         ▼
                   ┌───────────┐
                   │ Linear    │
                   │  32 → 1   │
                   └─────┬─────┘
                         │
                         ▼
                  Evaluation Score
            (+ = good for side-to-move)
```

### Feature Transformer

The Feature Transformer is the largest component (~5M of ~5M total parameters). It works like an embedding bag:

```python
output = bias + sum(weight[i] for i in active_indices)
```

- **Weight matrix**: 19,638 rows × 256 columns
- **Bias vector**: 256 elements
- **Input**: ~40 sparse indices (out of 19,638 possible)
- **Output**: 256-dimensional dense vector

The same weight matrix and bias are shared between STM and NSTM perspectives. Two separate outputs are produced (one per perspective) and concatenated into a 512-dimensional vector.

### Evaluation Network

Three small fully-connected layers that map the 512-dim feature vector to a single score:

| Layer | Input → Output | Parameters | Activation       |
|-------|---------------|------------|------------------|
| L1    | 512 → 32      | 16,416     | ClippedReLU [0,1] |
| L2    | 32 → 32       | 1,056      | ClippedReLU [0,1] |
| L3    | 32 → 1        | 33         | None (raw score)  |

Total evaluation network parameters: **17,505**

These layers are intentionally tiny. In a game engine, this part runs at every leaf node of the search tree — millions of times per second. Keeping it small is critical for playing strength.

### ClippedReLU

```
ClippedReLU(x) = clamp(x, 0.0, 1.0)
```

Unlike standard ReLU (which only clips at 0), ClippedReLU also caps at 1. This serves two purposes:

1. **Quantization**: When the trained model is deployed in a game engine, weights and activations are converted to integers (int8/int16) for speed. Bounded activations make this conversion straightforward — the [0, 1] range maps cleanly to [0, 127].
2. **Stability**: Prevents activations from growing unboundedly, which can cause numerical issues during training.

### Parameter Count Summary

| Component            | Parameters    | Percentage |
|----------------------|---------------|------------|
| Feature Transformer  | 5,027,584     | 99.7%      |
| Evaluation Network   | 17,505        | 0.3%       |
| **Total**            | **5,045,089** | 100%       |

---

## 5. Training Methodology

### Loss Function

The network is trained with a combined loss that matches **two targets**:

#### 1. Score Loss (Engine Evaluation Matching)

```
L_score = MSE(sigmoid(predicted), sigmoid(target_score))
```

- `target_score` comes from a strong engine's search (in scaled centipawns).
- Both values are passed through sigmoid, comparing in "win probability" space.
- Sigmoid compresses extreme scores, so the network focuses more on the critical ±1000cp range than on positions where one side is already totally winning.

#### 2. Result Loss (Game Outcome Prediction)

```
L_result = MSE(sigmoid(predicted), game_result)
```

- `game_result` ∈ {0.0, 0.5, 1.0} = {loss, draw, win} for the side-to-move.
- sigmoid(predicted) is interpreted as "probability of winning".
- This teaches the network long-term strategic understanding that even a strong engine's static evaluation might miss.

#### Combined Loss

```
L_total = λ_score × L_score + λ_result × L_result
```

Default weights: `λ_score = 1.0`, `λ_result = 0.5`. The score loss is weighted higher because engine evaluations are more precise than game results (which are noisy — a winning position can still lose due to mistakes later in the game).

### Score Scaling

Raw engine scores in centipawns are divided by a **scaling factor** (default: 600) before applying sigmoid:

```
scaled_score = raw_centipawns / 600
win_probability ≈ sigmoid(scaled_score)
```

This means:
- +600cp advantage → sigmoid(1.0) ≈ 0.73 → ~73% win probability
- +300cp advantage → sigmoid(0.5) ≈ 0.62 → ~62% win probability
- 0cp (equal) → sigmoid(0.0) = 0.50 → 50% win probability
- -600cp disadvantage → sigmoid(-1.0) ≈ 0.27 → ~27% win probability

### Optimizer

**Adam** with:
- Learning rate: 1e-3 (configurable)
- Weight decay: 1e-6 (light L2 regularization)
- Gradient clipping: max norm of 1.0 (prevents exploding gradients)

### Learning Rate Schedule

Two options:

- **Cosine Annealing** (default): LR smoothly decreases following a cosine curve from the initial value to near zero. This gives the model a natural "fine-tuning" phase in later epochs.
- **Step Decay**: LR is multiplied by a factor (gamma) every N epochs. For example, halved every 50 epochs.

### Training Data Pipeline

```
SFEN file (disk)
    ↓  IterableDataset (stream line by line)
Parse SFEN → board state
    ↓
Extract features → sparse indices (STM + NSTM)
    ↓
Scale score, package result
    ↓  DataLoader (batch + pad + multi-worker)
Padded batch tensors → GPU → Model
```

Key design decisions:
- **Streaming** (IterableDataset): Training files can be tens of GB. Streaming reads one line at a time without loading everything into memory.
- **Padding**: Different positions have different numbers of pieces, so sparse index tensors have different lengths. A custom `collate_fn` pads them with -1 to create uniform batches. The model ignores -1 indices.
- **Multi-worker**: Multiple processes read and parse data in parallel, keeping the GPU fed.

---

## 6. SFEN Data Format

### Position Notation

SFEN (Shogi Forsyth-Edwards Notation) is the standard text format for Shogi positions. It extends chess FEN for Shogi's unique features (promoted pieces, hand pieces).

#### Format

```
<board> <side_to_move> <hand_pieces> <move_number>
```

#### Board Field

Describes the 9×9 board, row by row from rank 1 (top) to rank 9 (bottom), separated by `/`.

- Uppercase = Black's pieces, lowercase = White's pieces
- Digits = consecutive empty squares
- `+` prefix = promoted piece

Piece characters:

| Char | Piece    | Char | Promoted Piece   |
|------|----------|------|------------------|
| P/p  | Pawn     | +P/+p | Tokin          |
| L/l  | Lance    | +L/+l | Promoted Lance |
| N/n  | Knight   | +N/+n | Promoted Knight|
| S/s  | Silver   | +S/+s | Promoted Silver|
| G/g  | Gold     | —     | —              |
| B/b  | Bishop   | +B/+b | Horse          |
| R/r  | Rook     | +R/+r | Dragon         |
| K/k  | King     | —     | —              |

#### Side to Move

- `b` = Black (Sente) to move
- `w` = White (Gote) to move

#### Hand Pieces

- `-` = no pieces in hand
- Otherwise: piece characters with optional count prefix
- Example: `S2P` = 1 Silver + 2 Pawns (in Black's hand, since uppercase)
- Example: `RB2s3p` = Black has Rook + Bishop, White has 2 Silvers + 3 Pawns

#### Example: Starting Position

```
lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
```

```
  9   8   7   6   5   4   3   2   1
┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ l │ n │ s │ g │ k │ g │ s │ n │ l │  Rank 1 (White)
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│   │ r │   │   │   │   │   │ b │   │  Rank 2
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ p │ p │ p │ p │ p │ p │ p │ p │ p │  Rank 3
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │   │  Rank 4
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │   │  Rank 5
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │   │  Rank 6
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ P │ P │ P │ P │ P │ P │ P │ P │ P │  Rank 7
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│   │ B │   │   │   │   │   │ R │   │  Rank 8
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ L │ N │ S │ G │ K │ G │ S │ N │ L │  Rank 9 (Black)
└───┴───┴───┴───┴───┴───┴───┴───┴───┘
  Hand: - (empty)    Side to move: Black    Move: 1
```

### Training Data Extension

ShogiDojo extends the standard SFEN with two additional fields for training:

```
sfen <position_sfen>	<score>	<result>
```

- **score**: Engine evaluation in centipawns (positive = good for side-to-move)
- **result**: Game outcome for side-to-move (`1.0` = win, `0.5` = draw, `0.0` = loss)

Fields are separated by tabs or spaces. The `sfen` prefix is optional.

---

## 7. Glossary

| Term                    | Definition |
|-------------------------|------------|
| **Accumulator**         | The dense vector produced by the Feature Transformer. Can be incrementally updated when pieces move (in the engine, not during training). |
| **Centipawn (cp)**      | Unit of evaluation. 100cp ≈ 1 Pawn advantage. Used by engines to express how good a position is. |
| **ClippedReLU**         | Activation function: `clamp(x, 0, 1)`. Bounded version of ReLU for quantization compatibility. |
| **Drop**                | Placing a captured piece from hand onto an empty square. Unique to Shogi. |
| **Feature Transformer** | The first (and largest) layer of NNUE. Maps sparse piece features to a dense vector using weight lookup and summation. |
| **Gote (後手)**          | White; the second player. |
| **HalfKAv2**            | The NNUE feature encoding scheme: two perspectives (Half), king-position-dependent weights (K), all pieces as features (A), version 2. |
| **Hand (持ち駒)**        | The pool of captured pieces a player can drop back onto the board. |
| **King Bucket**         | A grouping of king squares that share the same feature weights. Reduces the weight table size while preserving king-position-dependent evaluation. |
| **NNUE**                | Efficiently Updatable Neural Network (ƎUИИ). Invented by Yu Nasu for Shogi (2018). Designed for fast incremental evaluation during game-tree search. |
| **NSTM**                | Not-Side-To-Move. The player who is waiting (it's the opponent's turn). |
| **Promotion**           | Upgrading a piece when it enters the opponent's territory (ranks 7–9 for Black, ranks 1–3 for White). |
| **Quantization**        | Converting floating-point weights to integers (int8/int16) for faster inference in the game engine. |
| **Score Scaling**       | Dividing raw centipawn scores by a factor (default 600) to normalize them for the sigmoid function. |
| **Sente (先手)**         | Black; the first player. |
| **SFEN**                | Shogi Forsyth-Edwards Notation. Text format for describing Shogi positions, analogous to FEN in chess. |
| **STM**                 | Side-To-Move. The player whose turn it is. |
| **Thermometer Encoding** | Encoding a count N by activating indices 0 through N-1. Used for hand pieces. |

---

## References

- Yu Nasu, "Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi" (2018) — the original NNUE paper.
- [YaneuraOu](https://github.com/yaneurao/YaneuraOu) — a top Shogi engine that uses NNUE, and a common source of training data.
- [Stockfish NNUE](https://github.com/official-stockfish/Stockfish) — the chess engine that popularized NNUE outside of Shogi. Its training code (nnue-pytorch) inspired much of the modern NNUE training methodology.
