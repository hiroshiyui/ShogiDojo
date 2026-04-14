# Model Training Reference

This document explains key concepts behind NNUE training in ShogiDojo. For architecture details, see `CLAUDE.md`. For hyperparameter descriptions, see `configs/default.yaml`.

See also: [Stockfish NNUE PyTorch Wiki](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html) — comprehensive reference on NNUE architecture, quantization, and training techniques. Written for chess but the core concepts apply directly to Shogi.

---

## Perspective Handling During Training

### Q: Is perspective (player side) switching required during training?

**No.** The data pipeline normalizes everything to the side-to-move's (STM) perspective automatically. No explicit perspective switching is needed in the training loop.

### How It Works

The full data flow ensures color-agnostic training through four stages:

#### 1. Training Data Is Already STM-Relative

In `gensfen` (`tools/gensfen/src/types.rs`), both the search score and game result are recorded from the **side-to-move's perspective**:

- **Score**: positive = good for whoever is about to move
- **Result**: 1.0 = the side-to-move won, 0.5 = draw, 0.0 = the side-to-move lost

#### 2. Feature Extraction Normalizes Perspectives Automatically

`sfen_to_features()` in `src/shogi_dojo/features.py` checks who the side-to-move is and returns indices accordingly:

```python
if side_to_move == BLACK:
    return black_indices, white_indices   # STM=Black's perspective
else:
    return white_indices, black_indices   # STM=White's perspective
```

So `stm_indices` always means "the current player's view" and `nstm_indices` always means "the opponent's view", regardless of actual color.

#### 3. The Model Sees a Color-Agnostic Input

`FeatureTransformer.forward_dense()` in `src/shogi_dojo/model.py` concatenates:

```
[STM_features (256-dim) || NSTM_features (256-dim)] -> 512-dim
```

The network always interprets the first 256 dims as "my pieces" and the second 256 dims as "their pieces". It never knows or cares whether "I" am Black or White.

#### 4. Training Targets Match This Convention

In `src/shogi_dojo/trainer.py`, the loss functions compare the model output directly against the STM-relative score and result. No flipping is needed.

### Why This Design Works

This is the core insight of HalfKA: by normalizing everything to STM/NSTM, the network learns **one** evaluation function that works for both sides. A Pawn advancing toward the opponent is always good for "me", whether "me" is Black or White. Without this normalization, the network would need to learn symmetric patterns twice (once for each color), wasting model capacity.

---

## Dual Loss Function

Training optimizes two objectives simultaneously, combined with configurable weights:

```
total_loss = lambda_score * score_loss + lambda_result * result_loss
```

### Score Loss (sigmoid MSE)

```python
score_loss = MSE(sigmoid(predicted), sigmoid(target_score))
```

- **What it teaches**: Tactical accuracy. The network learns to match the engine's centipawn evaluation.
- **Why sigmoid**: Raw scores span a huge range (e.g., -10000 to +10000 centipawns). Sigmoid compresses them into (0, 1), so a 50cp difference near equality matters more than a 50cp difference when already winning by 2000cp. This reflects reality: the difference between +50 and +100 matters far more than +2000 vs +2050.

### Result Loss (sigmoid MSE against game outcome)

```python
result_loss = MSE(sigmoid(predicted), game_result)
```

- **What it teaches**: Strategic understanding. Positions that lead to wins should score high; positions that lead to losses should score low.
- **Why it matters**: The engine score reflects short-range tactical evaluation at a fixed search depth. The game result captures long-term factors (king safety, piece coordination, pawn structure) that a shallow search might miss.

### Choosing Lambda Weights

The defaults (`lambda_score=1.0`, `lambda_result=0.5`) prioritize tactical accuracy while still learning from game outcomes. Adjusting these trades off:

- **Higher `lambda_score`**: Network tracks the engine more closely. Better for tactical positions but may overfit to the engine's biases.
- **Higher `lambda_result`**: Network focuses on "which positions actually win." Better for strategic understanding but noisier (a single game result applies to all positions in that game).

---

## Score Scaling

Raw centipawn scores are divided by `score_scaling` (default: 600) before being fed to the loss function:

```python
scaled_score = raw_score / 600.0
```

This matters because both loss functions use `sigmoid()`. The sigmoid function is most sensitive near 0 and saturates (flattens out) for large inputs:

- `sigmoid(0.0) = 0.50` (even position)
- `sigmoid(1.0) = 0.73` (roughly +600cp with default scaling)
- `sigmoid(3.0) = 0.95` (saturated, +1800cp)

With `score_scaling=600`:
- A +600cp advantage maps to `sigmoid(1.0) = 0.73`, roughly a 73% win probability.
- A +300cp advantage maps to `sigmoid(0.5) = 0.62`.
- Positions beyond ~1800cp are treated as nearly won/lost (sigmoid saturates).

If the scaling is **too small** (e.g., 100), most positions saturate and the network cannot distinguish between slight and large advantages. If **too large** (e.g., 10000), all scores compress near 0.5 and the network struggles to learn.

---

## ClippedReLU (Why Not Standard ReLU)

All activations use `clamp(x, 0, 1)` instead of `max(x, 0)`:

- **Standard ReLU**: output in [0, +inf). Unbounded activations are fine for typical neural networks but problematic for NNUE.
- **ClippedReLU**: output in [0, 1]. Bounded activations are essential because NNUE models are quantized to 8-bit or 16-bit integers for fast inference in game engines. Bounded values make quantization straightforward (map [0, 1] to [0, 127]).

During training in PyTorch, ClippedReLU is simply `torch.clamp(x, 0.0, 1.0)`.

---

## Sparse Input and the -1 Padding Convention

Each position has a variable number of active features (~30-40 board pieces + hand pieces). To batch positions together efficiently, shorter index tensors are padded with **-1** as a sentinel value.

This convention is **load-bearing** and must stay in sync across two files:

1. **`dataset.py`** (`collate_fn`): Pads shorter tensors with -1 when creating batches.
2. **`model.py`** (`FeatureTransformer.forward_dense`): Clamps -1 to 0 for safe indexing, then zeros out the looked-up weight vectors at padding positions so they contribute nothing.

If either side changes the padding value or masking logic, training will silently produce wrong results.

---

## Training Data Generation (Iterative Loop)

NNUE training follows an iterative self-improvement loop:

```
Phase 1: Generate data with PST evaluator (material + piece-square tables)
            |
            v
Phase 2: Train NNUE on the generated data
            |
            v
Phase 3: Export trained NNUE weights -> regenerate data with NNUE evaluator
            |
            v
         Repeat Phase 2-3 until convergence
```

- **Phase 1** uses `gensfen --depth 4` with the built-in PST evaluator. The data quality is limited by the simple evaluation, but it bootstraps the process.
- **Phase 2** trains on whatever data is available. The NNUE learns to predict both the search scores and game outcomes.
- **Phase 3** uses `gensfen --depth 5 --model best_model.nnue` with the trained NNUE as the evaluator. Because the NNUE evaluates better than PST, the search plays stronger, producing higher-quality training data.

Each iteration produces stronger data, which trains a stronger network, which produces even stronger data.

### Random Opening Moves

Each self-play game begins with `random_moves` (default: 8) random legal moves before switching to engine play. This creates **opening diversity** in the training data — without it, every game would follow the same opening and the network would only learn about a narrow set of positions.

---

## Checkpointing and Resuming

### Periodic Checkpoints

Every `save_every` epochs (and always at the final epoch), a checkpoint is saved to `checkpoint_dir/epoch_NNNN.pt` containing:

- `model_state_dict`: All learned weights
- `optimizer_state_dict`: Adam optimizer state (momentum buffers, etc.)
- `epoch`: The epoch number
- `best_loss` / `best_epoch`: Best-model tracking state

### Best Model

Whenever the tracked loss (validation if available, else training) reaches a new minimum, the model is saved as `checkpoint_dir/best_model.pt`. This file is always overwritten with the latest best. After training completes, `best_model.pt` contains the single best weights for deployment.

### Resuming from Checkpoint

```bash
python train.py --resume checkpoints/epoch_0050.pt
```

This restores the model weights, optimizer state, and best-loss tracking. Training continues from the saved epoch.

---

## Early Stopping

If the tracked loss does not improve for `early_stopping_patience` consecutive epochs, training stops early. This prevents:

- Wasting compute on a converged model
- Overfitting when the model starts memorizing training data

Set `early_stopping_patience: 0` to disable and always train for the full number of epochs.

Example with `patience=20`: if the best loss was at epoch 50 and no improvement is seen by epoch 70, training stops at epoch 70. A final checkpoint is saved before stopping.

---

## Learning Rate Scheduling

Two schedulers are supported (configured via `lr_scheduler` in the YAML config):

### Cosine Annealing (Recommended)

```yaml
lr_scheduler: "cosine"
```

Smoothly decreases the learning rate following a cosine curve from the initial LR down to ~0 over all epochs. This is the standard choice for NNUE training because it allows the network to make large updates early (exploring the loss landscape) and fine-tune with small updates later.

### Step Decay

```yaml
lr_scheduler: "step"
lr_step_size: 50
lr_gamma: 0.5
```

Multiplies the LR by `gamma` every `step_size` epochs. For example, with the defaults above, LR is halved at epoch 50, 100, 150, etc. Less smooth than cosine but can be useful if you want to control exactly when the LR drops.

---

## Gradient Clipping

```yaml
clip_grad_norm: 1.0
```

If the total gradient norm exceeds this threshold after a backward pass, all gradients are scaled down proportionally. This prevents **exploding gradients** — a situation where a single unusual batch causes extremely large weight updates that destabilize the entire model.

Set to `0` to disable. The default of `1.0` is conservative and rarely needs adjustment.

---

## Measuring Model Strength

The companion tool `eval_match` (built alongside `gensfen`) plays a configurable number of self-play games between two evaluators and reports win/loss/draw counts, win rates, and an estimated Elo difference. Games are played in color-swapped pairs so that opening-color advantage cancels out.

### Baseline Comparison: Model vs. PST

```bash
./tools/gensfen/target/release/eval_match \
    --model-a checkpoints/model_v1.nnue \
    --games 200 --depth 4
```

Omitting `--model-b` makes Model A play against the built-in PST evaluator. This is the **absolute reference** — any reasonable NNUE should beat PST decisively (95%+ win rate). If it does not, something is wrong with training or with the exported `.nnue` file.

### Regression Comparison: Model vs. Model

```bash
./tools/gensfen/target/release/eval_match \
    --model-a model_v1.nnue --model-b model_v2.nnue \
    --games 200 --depth 4
```

Use this after every training iteration to confirm the new model is actually stronger. A typical healthy gain is **+20 to +50 Elo per iteration** in early bootstrap rounds, diminishing later.

### How Many Games Are Enough?

Elo estimates from small samples are noisy. As a rough guide:

- **100 games**: ±70 Elo confidence — only useful for large differences
- **200 games**: ±50 Elo — adequate for routine regression checks
- **1000 games**: ±20 Elo — needed to confirm small improvements

If two models are within ~30 Elo of each other, you need many games to be sure which is stronger.

---

## Improvement Strategy

When a trained model underperforms or you want to push it further, attack these levers in order. **Data quality dominates everything else early on** — do not tune hyperparameters or change architecture until data is no longer the bottleneck.

### 1. More and Better Training Data (Biggest Lever)

- **Quantity**: NNUE typically needs **100M–1B positions** to reach competitive strength. 100k positions is a sanity-check baseline; aim for at least 10M before judging the architecture.
- **Search depth**: Depth-4 PST data is noisy. After the first model trains, regenerate data with `--model your_v1.nnue --depth 6` — stronger evaluation at deeper search produces higher-quality labels.
- **Diversity**: Increase `--random_moves` (8 -> 12-16) so games cover more openings, not just the same standard lines.

### 2. Iterative Training (the NNUE Flywheel)

This is the **single most important workflow** in NNUE training:

```
PST -> generate data -> train v1 -> generate better data with v1
                                               |
                                               v
                                  train v2 -> generate even better data with v2 -> ...
```

Each generation should produce measurable Elo gains via `eval_match`. Stop iterating when gains plateau — typically after 5-10 generations. Each generation takes longer than the last (deeper search is slower), so plan accordingly.

### 3. Hyperparameter Tuning

Only after data quality is exhausted:

- **`lambda_result`** (currently 0.5): Try 0.3 (trust score more) or 0.7 (trust game outcome more). Compare via `eval_match`.
- **Learning rate schedule**: Cosine annealing is the standard choice; start at 1e-3 and decay over training.
- **Batch size**: Larger batches (8k-16k) often help NNUE convergence if GPU memory allows.
- **Epochs**: The early-stopping mechanism handles this automatically — usually no need to tune.

### 4. Architecture Changes (Last Resort)

Change architecture only after the iterative loop has plateaued:

- **More king buckets** (9 -> 16 or 32): finer per-king-position weights, more parameters.
- **Wider FeatureTransformer** (256 -> 512 or 1024): Stockfish uses 1024+. Big quality gain, big size cost.
- **Factorizer features**: virtual features that help generalization in early training, then folded into main weights at inference time. See "Feature Factorization" below.

### 5. Engine-Side Improvements (Separate from Training)

Stronger search makes the *same* NNUE play stronger games. But this is engine work (transposition tables, killer moves, late-move reductions in `gensfen`/an external engine), not training work — and it does not improve the evaluation function itself.

---

## Online Competition (Realistic Expectations)

A common question: can a trained NNUE compete against online players, the way AlphaGo competed against professional Go players?

**Short answer: not directly, and not as a like-for-like comparison.** Several reasons:

- **You have an evaluator, not an engine.** NNUE is just the evaluation function. Competitive online play needs a full search engine: alpha-beta with quiescence search, transposition tables, move ordering, time management, and the **USI protocol** (the standard for Shogi engine communication). The current `gensfen` search is teaching-quality alpha-beta — far below production engines like YaneuraOu or Elmo.
- **Strength gap**: Top Shogi engines reach **4400+ Elo**. A 5M-parameter HalfKAv2 trained on self-play data with depth-4 material seeding will land far below that. AlphaGo had massive compute, MCTS, and policy+value networks — a fundamentally different architecture.
- **Different problem domain**: AlphaGo was a research breakthrough that combined MCTS with deep policy/value networks. NNUE is an engineering optimization for fast CPU evaluation in classical alpha-beta engines. They are not comparable approaches.

### If You Want to Try Anyway

The realistic path:

1. Export the trained NNUE in a format compatible with **YaneuraOu** (HalfKAv2 is roughly compatible with their loader, with some conversion work).
2. Let YaneuraOu's mature search drive your evaluator.
3. Connect via USI to **floodgate** — the Japanese computer-Shogi server where amateur and professional engines compete. (This is the appropriate venue, not human-player servers like Shogi Wars.)

This is a multi-month project on top of solid training results. For now, `eval_match` is the right yardstick.

---

## Future Directions (from Stockfish NNUE Wiki)

The following techniques are documented in the [Stockfish NNUE PyTorch Wiki](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html) and could be adopted in ShogiDojo as the project matures:

### Feature Factorization

Virtual features (e.g., piece-only features without king position) can be added during early training to help the network converge faster. They relate the dense sub-features to the sparse HalfKA features, then their weights are folded into the main feature transformer weights after training by simple addition. This is especially helpful when training from scratch with limited data.

### Quantization for Engine Inference

To use a trained NNUE in a Shogi engine, weights must be quantized from float32 to integers:

- **Feature Transformer**: Scale weights/biases by 127 (int16 storage). Maps ClippedReLU output range [0, 1] to integer range [0, 127].
- **Hidden layers**: int8 weights scaled by 64, with int32 accumulation. Biases scaled by `input_scale * weight_scale`.
- **Output layer**: Additional scaling factor maps the network output to centipawn units.

The key constraint: ClippedReLU's bounded [0, 1] output makes this quantization straightforward, which is why we use it instead of standard ReLU.

### PSQT Output Buckets

Modern Stockfish architectures forward part of the feature transformer output directly to the final score (bypassing the hidden layers), acting as a learned piece-square table. This gives the network a "fast path" for basic material/positional evaluation without increasing hidden layer size.

### Quantmoid4

A piecewise quadratic approximation of `sigmoid(4x)` that can replace ClippedReLU in certain layers. It provides smoother gradients near the boundaries (0 and 1) while remaining cheap to compute in integer arithmetic.
