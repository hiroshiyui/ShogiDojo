# Bonanza / AlphaZero Network — Deployment Contract

This document is the **single source of truth** for what a consumer of an exported `bonanza.onnx` (or `bonanza.ts`) file must implement to actually use the network. The exported file contains only the neural network — the surrounding encoder, mask, and decoder logic must be reimplemented in the deployment language (Swift, Kotlin, JavaScript, Rust, etc.).

The contract does not depend on network size: `bonanza.onnx` (v1, 4×64) and `bonanza_v2.onnx` (v2, 10×128) have exactly the same inputs and outputs, so a consumer can swap one for the other.

It applies equally to checkpoints from `BonanzaTrainer` and `AlphaZeroTrainer` because they share the same `AlphaZeroNetwork` architecture and use the same I/O conventions.

The reference implementations are:

- Encoder / decoder / mask: [`src/shogi_dojo/alphazero/encoding.py`](../../src/shogi_dojo/alphazero/encoding.py)
- Value sign-flip on backup: [`src/shogi_dojo/alphazero/mcts.py`](../../src/shogi_dojo/alphazero/mcts.py)
- Network shape: [`src/shogi_dojo/alphazero/network.py`](../../src/shogi_dojo/alphazero/network.py)

If anything below disagrees with `encoding.py`, **the source is correct and this document is wrong** — please file a doc fix.

---

## 1. The .onnx file: I/O contract

The exporter (`scripts/alphazero/export_onnx.py`) writes a model with:

| Tensor | Name | Shape | Dtype | Notes |
|---|---|---|---|---|
| Input | `board` | `(B, 45, 9, 9)` | `float32` | Encoded position planes (see §3) |
| Output | `policy_logits` | `(B, 139, 9, 9)` | `float32` | Raw logits over the 9×9×139 action space (see §4) |
| Output | `value` | `(B, 1)` | `float32` | Scalar in `(-1, +1)` from STM's perspective (see §5) |

The batch axis (`B`) is dynamic — the same file handles single-position mobile inference and large-batch server inference.

The network does **not** apply softmax, legal-move masking, or any decoding. All of that is the consumer's responsibility.

---

## 2. Side-to-move convention (read this before §3 and §4)

The network was trained to always see the world **from the side-to-move's perspective**, with their own pieces at the bottom of the input planes and "forward = decreasing rank index."

This means:

- When **Black is to move**, the canonical board frame is used directly — Black's pieces sit on ranks 7–9 (rows 6, 7, 8 in 0-indexed coordinates), facing up.
- When **White is to move**, the board is **rotated 180°** before encoding — every square `(rank, file)` becomes `(8 − rank, 8 − file)`, and "own" / "opponent" piece colors are swapped. The network never sees a White's-eye-view directly.

The same flip applies to both **inputs** (position planes) and **outputs** (policy index decoding).

The value head's output is also from STM's perspective: `+1` means the side-to-move is winning, `−1` means losing — regardless of who actually plays Black or White. See §5.

```
Coordinate convention used throughout this document:

  rank index y: 0 = top of board (opponent's back rank from STM's view)
                8 = bottom of board (STM's back rank)
  file index x: 0 = left, 8 = right
  square s    : y * 9 + x  (so s ∈ [0, 81))
```

When you flip:

```
flip_yx(y, x) = (8 - y, 8 - x)
flip_color(c) = 1 - c        // BLACK=0, WHITE=1
```

---

## 3. Position encoder (45 input planes)

Build a `(45, 9, 9)` float32 tensor, all zeros to start. Then fill it as follows.

### 3.1 Plane layout summary

| Channels | Content | Encoding |
|---|---|---|
| `0..13`   | Own pieces, one plane per piece type | Binary 0/1 at each square |
| `14..27`  | Opponent pieces, one plane per piece type (same order) | Binary 0/1 at each square |
| `28..34`  | Own hand piece counts (P, L, N, S, G, B, R) | Constant fill: `count / max_count` |
| `35..41`  | Opponent hand piece counts (same order) | Constant fill: `count / max_count` |
| `42`      | Repetition count | Constant fill: `min(rep, 3) / 3.0` |
| `43`      | Side-to-move plane | Constant fill: always `1.0` |
| `44`      | Move number | Constant fill: `min(move_number, 200) / 200.0` |

Total: `14 + 14 + 7 + 7 + 1 + 1 + 1 = 45` channels.

### 3.2 Per-piece planes (channels 0–27)

The 14 piece types, in plane order:

| Plane offset | Piece | Symbol | Note |
|---:|---|---|---|
| 0  | Pawn        | 歩 (P)   | |
| 1  | Lance       | 香 (L)   | |
| 2  | Knight      | 桂 (N)   | |
| 3  | Silver      | 銀 (S)   | |
| 4  | Gold        | 金 (G)   | |
| 5  | Bishop      | 角 (B)   | |
| 6  | Rook        | 飛 (R)   | |
| 7  | King        | 玉/王 (K) | |
| 8  | Pro-Pawn    | と       | promoted Pawn |
| 9  | Pro-Lance   | 成香     | promoted Lance |
| 10 | Pro-Knight  | 成桂     | promoted Knight |
| 11 | Pro-Silver  | 成銀     | promoted Silver |
| 12 | Horse       | 馬       | promoted Bishop |
| 13 | Dragon      | 龍       | promoted Rook |

For each piece on the board:

```
flip = (side_to_move == WHITE)
for each (piece_type, color, square) on the board:
    (y, x) = square_to_yx(square)
    if flip:
        (y, x) = flip_yx(y, x)
        owner_is_stm = (color == WHITE)
    else:
        owner_is_stm = (color == BLACK)

    plane_offset = piece_type            // 0..13 from the table above
    base = (owner_is_stm) ? 0 : 14
    tensor[base + plane_offset, y, x] = 1.0
```

### 3.3 Hand piece planes (channels 28–41)

Hand pieces are encoded as a **broadcast normalized count** — the same scalar fills the entire 9×9 plane.

The 7 droppable piece types (in plane order):

| Plane offset | Piece | Max count |
|---:|---|---:|
| 0 | Pawn   | 18 |
| 1 | Lance  | 4  |
| 2 | Knight | 4  |
| 3 | Silver | 4  |
| 4 | Gold   | 4  |
| 5 | Bishop | 2  |
| 6 | Rook   | 2  |

```
own_color = flip ? WHITE : BLACK
opp_color = 1 - own_color

for i in 0..6:
    pt = HAND_PIECE_TYPES[i]
    max = HAND_PIECE_MAX[pt]                     // from table above
    own_norm = hand[own_color][i] / max          // float in [0, 1]
    opp_norm = hand[opp_color][i] / max
    fill_plane(tensor, 28 + i, own_norm)
    fill_plane(tensor, 35 + i, opp_norm)
```

### 3.4 Scalar planes (channels 42–44)

```
fill_plane(tensor, 42, min(repetition_count, 3) / 3.0)
fill_plane(tensor, 43, 1.0)                       // always 1.0
fill_plane(tensor, 44, min(move_number, 200) / 200.0)
```

`repetition_count` is "how many times has this exact position previously occurred in the game" — `0` if you are not tracking it. The network was trained on the wdoor archive, which has occasional repetitions but is dominated by the `0` case, so passing `0` everywhere is acceptable for casual deployments.

`move_number` is the 1-indexed full move counter (the 4th field of an SFEN string).

---

## 4. Policy decoder (139 output planes)

The policy head outputs `(139, 9, 9)` raw logits. To turn this into a move:

1. Build a `(139, 9, 9)` boolean **legal-move mask** — exactly the indices that correspond to legal moves at the current position.
2. Set illegal logits to a large negative number (`-1e9` is what training used).
3. Apply softmax to get probabilities.
4. Pick a move (argmax for greedy play, or sample for exploration).
5. Decode the chosen index back to a Shogi move (file/rank/promote/drop).

The expensive part is step 1 — it requires a **Shogi rule engine** in the deployment language (legal moves, pinned-piece detection, drop legality including pawn-drop mate, last-rank promotion forcing, etc.). Use an existing library:

- Rust: [`shogi`](https://crates.io/crates/shogi) crate
- JavaScript: [`Shogi.js`](https://github.com/na2hiro/Shogi.js)
- Swift: port from python-shogi or shogi-rs (~1k lines)
- Kotlin: port similarly, or call shogi-rs via JNI

### 4.1 Index decomposition

Every flat policy index `i ∈ [0, 11_259)` decomposes as:

```
plane = i // 81
sq    = i % 81
y     = sq // 9
x     = sq % 9
```

Inverse (compose):

```
i = plane * 81 + (y * 9 + x)
```

`(y, x)` always refers to coordinates in the **STM-perspective frame** (after the flip from §2). For board moves it is the **source** square; for drops it is the **destination** square.

### 4.2 Plane layout

| Plane range | Count | Meaning |
|---|---:|---|
| `0..63`    | 64 | Sliding "queen-like" board move from `(y, x)` |
| `64..65`   |  2 | Knight move from `(y, x)` |
| `66..129`  | 64 | Same as 0..63 but **with promotion** |
| `130..131` |  2 | Same as 64..65 but **with promotion** |
| `132..138` |  7 | Drop of a held piece **to** `(y, x)` |

Total: `64 + 2 + 64 + 2 + 7 = 139` planes.

### 4.3 Sliding "queen" planes (0..63 and 66..129)

A sliding move travels in one of 8 directions for 1..8 squares:

```
queen_plane = direction_index * 8 + (distance - 1)        // 0..63
```

Direction ordering (in STM-perspective frame, where forward = `−y`):

| `direction_index` | Name | `(dy, dx)` per step |
|---:|---|---|
| 0 | N  (forward)    | `(-1,  0)` |
| 1 | NE              | `(-1, +1)` |
| 2 | E               | `( 0, +1)` |
| 3 | SE              | `(+1, +1)` |
| 4 | S  (backward)   | `(+1,  0)` |
| 5 | SW              | `(+1, -1)` |
| 6 | W               | `( 0, -1)` |
| 7 | NW              | `(-1, -1)` |

`distance` ∈ `1..8`. The destination is `(from_y + dy * distance, from_x + dx * distance)`, in the STM-perspective frame.

For the **promoting** variant, add 66 to the plane index:

```
promoting_queen_plane = 66 + (direction_index * 8 + (distance - 1))     // 66..129
```

### 4.4 Knight planes (64..65 and 130..131)

The Shogi knight is unusual — it only moves forward, jumps over intervening pieces, and lands two ranks ahead and one file to either side. In the STM-perspective frame:

| `knight_plane` | Offset `(dy, dx)` | Meaning |
|---:|---|---|
| 64  | `(-2, -1)` | Forward-left knight jump  |
| 65  | `(-2, +1)` | Forward-right knight jump |

Promoting variants:

| `promoting_knight_plane` | Offset |
|---:|---|
| 130 | `(-2, -1)` + promote |
| 131 | `(-2, +1)` + promote |

### 4.5 Drop planes (132..138)

A drop has no source square — `(y, x)` is the **destination** in the STM-perspective frame.

| `plane` | Piece dropped |
|---:|---|
| 132 | Pawn   |
| 133 | Lance  |
| 134 | Knight |
| 135 | Silver |
| 136 | Gold   |
| 137 | Bishop |
| 138 | Rook   |

Drops cannot promote — there is no "promoting drop" plane.

### 4.6 Decoding pseudocode

```
function decode_index(index, side_to_move):
    plane = index // 81
    sq    = index % 81
    y     = sq // 9
    x     = sq % 9

    # --- Drop ---
    if plane >= 132:
        drop_offset = plane - 132
        drop_piece  = HAND_PIECE_TYPES[drop_offset]   // see §3.3 table
        if side_to_move == WHITE:
            (y, x) = flip_yx(y, x)
        return Move(from = null, to = yx_to_square(y, x),
                    drop = drop_piece, promote = false)

    # --- Board move ---
    promote    = (plane >= 66)
    geom_plane = promote ? (plane - 66) : plane

    if geom_plane >= 64:
        # Knight
        offset = KNIGHT_OFFSETS[geom_plane - 64]      // see §4.4
        (dy, dx) = offset
    else:
        # Queen-like
        direction_index = geom_plane // 8
        distance        = (geom_plane % 8) + 1
        (udy, udx)      = QUEEN_DIRECTIONS[direction_index]   // see §4.3
        (dy, dx)        = (udy * distance, udx * distance)

    to_y = y + dy
    to_x = x + dx
    if side_to_move == WHITE:
        (from_y, from_x) = flip_yx(y, x)
        (to_y,   to_x)   = flip_yx(to_y, to_x)
    else:
        (from_y, from_x) = (y, x)

    return Move(from    = yx_to_square(from_y, from_x),
                to      = yx_to_square(to_y, to_x),
                drop    = null,
                promote = promote)
```

---

## 5. Value head — perspective and sign-flip

The value output is a single float in `(-1, +1)` after `tanh`:

- `+1.0` ⇒ STM is winning
- `0.0`  ⇒ position is balanced
- `-1.0` ⇒ STM is losing

This is **always from the side-to-move's perspective at the position you fed in**, not from a fixed (Black or White) perspective. Concrete example:

> If it's Black's turn and the network outputs `value = +0.7`, that means *Black* is favored by ~0.7. If you feed in the *next* position (after Black plays a move and it becomes White's turn) and the network outputs `value = +0.6`, that means *White* is now favored by ~0.6 — which is consistent with the previous evaluation, because Black's strong position has flipped to White's perspective and lost a small amount of edge from the move played.

### 5.1 The sign-flip rule (only matters inside a search)

If you build a search on top of the network (MCTS, alpha-beta, even simple "look one move ahead"), you must flip the sign of the value at every ply transition, because the perspective alternates:

```
my_eval_after_move  = -opponent_eval_after_their_response
```

Equivalently, for MCTS backup:

```
function backup(path, leaf_value):
    v = leaf_value     # always from leaf's STM perspective
    for node in reverse(path):
        v = -v         # flip on every ply
        node.total_value += v
        node.visit_count += 1
```

For greedy play **without search** (pick the argmax move from the policy and play it), you can ignore the value head entirely — only the policy matters.

The reference is [`mcts.py::MCTS._simulate`](../../src/shogi_dojo/alphazero/mcts.py); CLAUDE.md flags this as a load-bearing invariant ("Perspective sign-flip on MCTS backup") with an explicit unit test that catches getting the sign wrong.

---

## 6. End-to-end inference (one position)

Pseudocode for "given a Shogi position, ask the network for a move":

```
function pick_move(position):
    # 1. Encode
    board_tensor = build_45x9x9_planes(position)        // §3
    board_tensor = reshape to (1, 45, 9, 9)             // batch axis

    # 2. Run the network
    (policy_logits, value) = onnx_session.run(board_tensor)
    policy_logits = policy_logits.squeeze(0)            // (139, 9, 9)
    value         = value.item()                        // scalar in (-1, +1)

    # 3. Build legal mask (requires a Shogi rule engine!)
    legal_moves = position.legal_moves()
    mask = zeros((139, 9, 9), dtype=bool)
    for move in legal_moves:
        idx = encode_move(move, position.side_to_move)  // §4 inverse
        plane, sq = divmod(idx, 81)
        y, x      = divmod(sq, 9)
        mask[plane, y, x] = true

    # 4. Mask + softmax
    flat_logits = policy_logits.flatten()               // (11259,)
    flat_mask   = mask.flatten()
    flat_logits[!flat_mask] = -1e9
    probs = softmax(flat_logits)

    # 5. Pick
    chosen_index = argmax(probs)                        // or sample
    return decode_index(chosen_index, position.side_to_move), value
```

For a "show top-3 candidates" hint mode (the natural sensei UX), step 5 becomes:

```
top_k_indices = top_k(probs, k=3)
return [decode_index(i, position.side_to_move) for i in top_k_indices], value
```

---

## 7. Quick reference

```
Colors:
  BLACK=0  (Sente, 先手, moves first)
  WHITE=1  (Gote,  後手, moves second)

Board:
  9 ranks × 9 files = 81 squares, indexed row-major from top-left:
    square = y * 9 + x
    y, x   = square / 9, square % 9

Hand piece order (used in §3.3 and §4.5 — same list, same order):
  [Pawn, Lance, Knight, Silver, Gold, Bishop, Rook]

Per-piece plane order (§3.2):
  [Pawn, Lance, Knight, Silver, Gold, Bishop, Rook,
   King,
   Pro-Pawn, Pro-Lance, Pro-Knight, Pro-Silver, Horse, Dragon]
```

**Important note for readers of the Python source:** the integer piece-type IDs in `src/shogi_dojo/nnue/constants.py` (`PAWN=0`, …, `DRAGON=12`, `KING=13`) are **not** the plane offsets in §3.2. They differ for King and the promoted pieces. The encoder maps source-code constants to plane offsets via the `ALL_PIECE_TYPES_ORDERED` list in `encoding.py`:

| Source-code constant | Integer value | Plane offset (§3.2) |
|---|---:|---:|
| `KING`        | 13 | 7  |
| `PRO_PAWN`    |  7 | 8  |
| `PRO_LANCE`   |  8 | 9  |
| `PRO_KNIGHT`  |  9 | 10 |
| `PRO_SILVER`  | 10 | 11 |
| `HORSE`       | 11 | 12 |
| `DRAGON`      | 12 | 13 |

When porting, **use the plane offsets in §3.2 directly** — the source-code constants are an internal artifact of NNUE handling kings via separate buckets, and have no significance in the deployed model.

Also: this repository's piece-type IDs are **not** the same as python-shogi's IDs (which are 1-indexed and use a different ordering). If you happen to be using python-shogi-style IDs in your target language's rule library, you'll need a small translation table.

---

## 8. Verifying your port

The simplest correctness check:

1. Pick a known SFEN — e.g. the standard starting position, `lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1`.
2. Run it through the **Python** encoder (`encode_position` in `encoding.py`) and dump the 45×9×9 tensor. Spot-check a few channels.
3. Build the same tensor in your target language using §3 above. Compare element-wise — they must match exactly.
4. Feed both into ONNX Runtime and confirm the outputs match within `1e-4`.

If step 3 disagrees, the mistake is almost always in the **flip** (§2) — most commonly: forgetting to flip "own" / "opponent" when STM is White, or confusing flipping the *board* with flipping the *piece colors* (you must do both).

Once that passes, you can build the rest (mask, decoder, search) on top with confidence that the input contract is correct.
