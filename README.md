# Shogi Dojo - Computer Shogi Studies Research Group

Academic research resources for studying neural network evaluation in Shogi (Japanese Chess), covering three paradigms:

- **NNUE (HalfKAv2)** — a sparse, incrementally-updatable evaluation network (~5M params) designed for real-time alpha-beta search. The primary pipeline: generate data with a Rust self-play tool, train with PyTorch, export quantized weights.
- **AlphaZero-style** — a deep residual policy/value network (~340K params, survey-sized) trained purely from self-play via MCTS. An educational reference implementation with comprehensive documentation.
- **Bonanza-method** — supervised learning from professional Shogi game records (kifu). The simplest of the three paradigms (no MCTS, no self-play, just a dataset and a training loop). Reuses the AlphaZero network architecture and can warm-start AlphaZero self-play.

## What is NNUE?

NNUE is a neural network architecture designed for fast board-game evaluation. It was originally invented for Shogi by Yu Nasu in 2018. The key insight: instead of evaluating the entire board from scratch every time, NNUE uses an "accumulator" that can be **incrementally updated** when a single piece moves, making it fast enough for real-time game-tree search (millions of evaluations per second).

For a deeper dive into the architecture, see [doc/nnue/NNUE_Reference.md](doc/nnue/NNUE_Reference.md).
For training concepts, see [doc/nnue/Model_Training_Reference.md](doc/nnue/Model_Training_Reference.md).

## Project Structure

The repository is organized so that each evaluation paradigm lives under its
own `nnue/` / `alphazero/` subdirectory. Code, configs, scripts, tests, and
docs all follow the same split.

```
ShogiDojo/
├── train_nnue.py                # CLI entry point for NNUE training
├── train_alphazero.py           # CLI entry point for AlphaZero training
├── pyproject.toml               # Project metadata and dependencies
├── configs/
│   ├── nnue/
│   │   ├── default.yaml         # Default NNUE hyperparameters (~5M params)
│   │   ├── small.yaml           # Small model for quick experiments (~640K params)
│   │   └── large.yaml           # Large model for maximum strength (~20M params)
│   └── alphazero/
│       ├── tiny.yaml            # Smoke-test config (runs in minutes on CPU)
│       └── default.yaml         # Laptop-scale training (hours)
├── data/                        # Place training data here (.sfen files)
├── src/shogi_dojo/
│   ├── nnue/
│   │   ├── constants.py         # Board, piece, and feature-space constants
│   │   ├── features.py          # SFEN parsing and HalfKAv2 feature extraction
│   │   ├── model.py             # Neural network (FeatureTransformer + eval net)
│   │   ├── dataset.py           # Streaming data loader for SFEN files
│   │   └── trainer.py           # Training loop, loss functions, checkpointing, TensorBoard
│   └── alphazero/
│       ├── encoding.py          # Board → 45-ch tensor; Move ↔ 9×9×139 policy index
│       ├── network.py           # ResNet + policy/value heads (~336K params)
│       ├── mcts.py              # Game-agnostic PUCT MCTS
│       ├── rules.py             # python-shogi adapter for legal moves + terminal detection
│       ├── self_play.py         # Self-play game driver + replay buffer
│       └── trainer.py           # Iteration loop (self-play → buffer → train → checkpoint)
├── scripts/
│   └── nnue/
│       ├── export_nnue.py           # Export trained model to binary NNUE (float32)
│       ├── export_nnue_quantized.py # Export quantized NNUE (int8/int16)
│       ├── train_loop.py            # Iterative training loop orchestration
│       ├── validate_sfen.py         # Training data validation and statistics
│       └── benchmark.py             # Inference and training throughput benchmarks
├── tests/
│   ├── nnue/                    # NNUE test suite (152 tests)
│   └── alphazero/               # AlphaZero test suite (177 tests)
├── tools/
│   └── nnue/
│       └── gensfen/             # Rust tool: training data generation + model evaluation
│           ├── src/main.rs           # gensfen binary (data generation)
│           └── src/bin/eval_match.rs # eval_match binary (model strength evaluation)
├── .github/workflows/ci.yml    # CI: Python (ruff + pytest) and Rust (fmt + clippy + test)
├── doc/
│   ├── nnue/
│   │   ├── NNUE_Reference.md           # Architecture deep-dive
│   │   └── Model_Training_Reference.md # Training concepts and techniques
│   ├── alphazero/
│   │   ├── AlphaZero_Reference.md           # Architecture, MCTS, encoding, glossary
│   │   ├── AlphaZero_Training_Reference.md  # Training loop, hyperparameters, failure modes
│   │   ├── AlphaZero_vs_NNUE.md             # Operational comparison
│   │   └── AlphaZero_Reading_List.md        # Annotated bibliography
│   └── TODO.md                  # Project roadmap (shared across paradigms)
└── virtualenv/                  # Python virtual environment
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.0
- NumPy, PyYAML
- python-shogi (for the AlphaZero survey's legal-move generation)
- Rust toolchain (for gensfen/eval_match tools)

## Setup

```bash
git clone <repo-url>

# Activate the virtual environment
source virtualenv/bin/activate

# Install in editable mode (add [dev] for tests, [viz] for TensorBoard)
pip install -e .
pip install -e ".[dev]"     # pytest + ruff
pip install -e ".[viz]"     # tensorboard
```

## Quick Start (End-to-End)

```bash
# 1. Build the Rust tools
cd tools/nnue/gensfen && cargo build --release && cd ../../..

# 2. Generate training data (~100K positions via self-play)
mkdir -p data
./tools/nnue/gensfen/target/release/gensfen \
    --depth 4 --threads 4 --positions 100000 --output data/train.sfen

# 3. Train the NNUE model
python train_nnue.py

# 4. Export the trained model
python scripts/nnue/export_nnue.py checkpoints/best_model.pt model.nnue
```

### Iterative Training Loop

For stronger models, use the automated iterative loop — each iteration generates better data using the previously trained model:

```bash
python scripts/nnue/train_loop.py --iterations 3 --positions 100000
```

This runs: gensfen (PST eval) -> train -> export -> gensfen (NNUE eval) -> train -> export -> ...

See `python scripts/nnue/train_loop.py --help` for all options.

## Generating Training Data

Training data is generated by the `gensfen` tool (`tools/nnue/gensfen/`), a Rust program that plays Shogi games against itself and records positions.

### Building gensfen

Requires the [Rust toolchain](https://rustup.rs/).

```bash
cd tools/nnue/gensfen && cargo build --release
```

The release profile enables LTO and `target-cpu=native` (see `tools/nnue/gensfen/.cargo/config.toml`) for ~20% faster NNUE inference. The resulting binary is **non-portable** — only run it on a machine with at least the same CPU feature set as the build host. For portable CI builds:

```bash
RUSTFLAGS="" cargo build --release
```

### Running gensfen

```bash
./tools/nnue/gensfen/target/release/gensfen \
    --depth 4 --threads 4 --positions 1000000 --output data/train.sfen
```

| Option | Default | Description |
|--------|---------|-------------|
| `--depth` | `4` | Alpha-beta search depth. Higher = better quality, slower. |
| `--threads` | `4` | Parallel game threads. |
| `--positions` | (required) | Total positions to generate. |
| `--output` | `output.sfen` | Output file path (text SFEN). |
| `--random-moves` | `8` | Random opening moves for diversity. |
| `--model` | (none) | Path to `.nnue` model file for NNUE-powered self-play. |

Without `--model`, gensfen uses the built-in PST evaluator (material + piece-square tables). With `--model`, it uses the trained NNUE for stronger self-play, producing higher-quality training data.

#### Picking a Depth

`gensfen` runtime grows roughly 5× per extra ply at NNUE inference cost. A position is only written *after the whole game it belongs to finishes*, so the per-game wall time below is also the time before any output appears.

Indicative numbers on a 6-core CPU (LTO + native build, `--threads 8`, NNUE evaluator):

| `--depth` | Per-game wall time | 100k-position ETA |
|-----------|-------------------:|------------------:|
| 3         | ~3 min             | ~3-4 hours        |
| 4         | ~30 min            | overnight (~15-20 h) |
| 5         | ~3 hours           | ~3-4 days         |
| 6         | ~15 hours          | weeks             |

For limited-resource setups, see ["Limited Resources / Constrained Search Depth"](doc/nnue/Model_Training_Reference.md#0-limited-resources--constrained-search-depth) in the training reference — iterating at depth 4-5 typically beats a single shot at depth 6+.

### Output Format

One position per line:

```
sfen <board> <side> <hand> <move#>\t<score>\t<result>
```

| Field     | Description                                               | Example                    |
|-----------|-----------------------------------------------------------|----------------------------|
| `board`   | SFEN board string (pieces on the 9x9 grid)                | `lnsgkgsnl/1r5b1/...`     |
| `side`    | Side to move: `b` (Black/Sente) or `w` (White/Gote)       | `b`                        |
| `hand`    | Pieces in hand (`-` if none)                               | `S2P` (1 Silver, 2 Pawns)  |
| `move#`   | Move number                                               | `1`                        |
| `score`   | Engine evaluation in centipawns (positive = good for STM)  | `100`                      |
| `result`  | Game result for STM: `1.0` = win, `0.5` = draw, `0.0` = loss | `1.0`                  |

### Validating Training Data

```bash
python scripts/nnue/validate_sfen.py data/train.sfen
python scripts/nnue/validate_sfen.py data/*.sfen --check-duplicates --summary
```

Reports malformed lines, score distribution, result balance (W/D/L), and duplicates.

### Alternative Data Sources

You can also use pre-generated datasets from [Hugging Face (nodchip/shogi_hao_depth9)](https://huggingface.co/datasets/nodchip/shogi_hao_depth9) (may require format conversion).

### Prepare Data Files

Split your data into training and (optionally) validation sets in `data/`:

```
data/
├── train.sfen    # Training positions (required)
└── val.sfen      # Validation positions (optional)
```

## Training

```bash
# Train with default configuration (~5M params)
python train_nnue.py

# Use a smaller model for quick experiments (~640K params)
python train_nnue.py --config configs/nnue/small.yaml

# Use a larger model for maximum strength (~20M params)
python train_nnue.py --config configs/nnue/large.yaml

# Resume from a checkpoint
python train_nnue.py --resume checkpoints/epoch_0050.pt
```

Training logs are printed to the terminal. If `log_dir` is set in the config (default: `runs`), metrics are also logged to TensorBoard:

```bash
tensorboard --logdir runs
```

Checkpoints are saved to `checkpoints/` every 10 epochs (configurable). The best model is always saved as `checkpoints/best_model.pt`.

## Exporting Models

### Float32 Export (for gensfen iterative training)

```bash
python scripts/nnue/export_nnue.py checkpoints/best_model.pt model.nnue
```

### Quantized Export (for game engine integration)

```bash
python scripts/nnue/export_nnue_quantized.py checkpoints/best_model.pt model_q.nnue
```

Exports int16 (Feature Transformer, scale=127) and int8 (hidden layers, scale=64) weights following the [Stockfish NNUE convention](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html).

## Evaluating Model Strength

The `eval_match` tool plays games between two models and reports win rates and Elo difference:

```bash
# Build
cd tools/nnue/gensfen && cargo build --release && cd ../../..

# Model vs. PST baseline
./tools/nnue/gensfen/target/release/eval_match \
    --model-a model.nnue --games 100 --depth 4

# Model vs. model
./tools/nnue/gensfen/target/release/eval_match \
    --model-a model_v2.nnue --model-b model_v1.nnue --games 200 --depth 4
```

Games are played in color-swapped pairs for fairness. Output includes win/loss/draw counts and estimated Elo difference.

## Benchmarking

```bash
python scripts/nnue/benchmark.py
python scripts/nnue/benchmark.py --config configs/nnue/small.yaml --device cpu
```

Measures inference speed (positions/sec), training throughput (samples/sec), parameter count, and GPU memory usage.

## Configuration

All hyperparameters are set in YAML config files. Three variants are provided:

| Config | Parameters | Use Case |
|--------|-----------|----------|
| `configs/nnue/default.yaml` | ~5M | Standard training |
| `configs/nnue/small.yaml` | ~640K | Quick experiments, debugging |
| `configs/nnue/large.yaml` | ~20M | Maximum strength |

Key options in the default config:

```yaml
data:
  batch_size: 8192       # Positions per batch (reduce if OOM)
  num_workers: 4         # Parallel data loading processes

model:
  ft_out_dim: 256        # Feature transformer width (per perspective)
  l1_dim: 32             # Eval network hidden layer 1
  l2_dim: 32             # Eval network hidden layer 2

training:
  epochs: 200
  lr: 1.0e-3             # Adam learning rate
  lr_scheduler: "cosine" # "cosine" or "step"
  lambda_score: 1.0      # Weight for engine-score loss
  lambda_result: 0.5     # Weight for game-result loss
  score_scaling: 600.0   # Centipawn normalization factor
  save_every: 10         # Checkpoint frequency (epochs)
  log_dir: "runs"        # TensorBoard log directory
```

See `configs/nnue/default.yaml` for the full list with detailed comments.

## Model Architecture

ShogiDojo uses the **HalfKAv2** NNUE architecture (~5M parameters):

```
Sparse Input (19,638 possible features, ~40 active per position)
    |
Feature Transformer (shared weights, sparse lookup + sum)
    |-- STM perspective  -> 256-dim dense vector
    +-- NSTM perspective -> 256-dim dense vector
    |
Concatenate -> 512-dim
    | ClippedReLU
Linear 512 -> 32
    | ClippedReLU
Linear 32 -> 32
    | ClippedReLU
Linear 32 -> 1  ->  Evaluation Score
```

- **STM** = Side-To-Move (the player whose turn it is)
- **NSTM** = Not-Side-To-Move (the opponent)
- **ClippedReLU** = `clamp(x, 0, 1)` — bounded activations for quantization compatibility

For a full explanation of HalfKAv2, king buckets, feature encoding, and loss functions, see [doc/nnue/NNUE_Reference.md](doc/nnue/NNUE_Reference.md).

## Disk Space Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Training data (text SFEN) | **5-100+ GB** | 100M positions ~ 5 GB |
| Single checkpoint | ~19 MB | Model weights + optimizer state |
| All checkpoints (default) | ~385 MB | 20 files (save every 10 x 200 epochs) |
| gensfen Rust tool | ~1 MB | Compiled binary |

| Scale | Positions | Training Data | Total Disk |
|-------|-----------|--------------|------------|
| Quick test | 1M | ~50 MB | **~1 GB** |
| Small run | 100M | ~5 GB | **~7 GB** |
| Full training | 1B | ~50 GB | **~55 GB** |

## AlphaZero Survey

A parallel research survey of the AlphaZero family of game-playing systems. The survey is *educational* — it does not aim to produce a competitive engine. Docs lead, code follows, everything is beginner-friendly.

### Documentation

| Document | What it covers |
|---|---|
| [doc/alphazero/AlphaZero_Reference.md](doc/alphazero/AlphaZero_Reference.md) | Architecture, MCTS+PUCT, Shogi encoding, history, glossary, references (~650 lines) |
| [doc/alphazero/AlphaZero_Training_Reference.md](doc/alphazero/AlphaZero_Training_Reference.md) | Self-play loop, replay buffer, hyperparameters, failure modes, resource estimates (~800 lines) |
| [doc/alphazero/AlphaZero_vs_NNUE.md](doc/alphazero/AlphaZero_vs_NNUE.md) | Operational comparison: inference cost, training compute, deployment, when-to-use-which |
| [doc/alphazero/AlphaZero_Reading_List.md](doc/alphazero/AlphaZero_Reading_List.md) | Annotated bibliography: core papers, open-source engines, tutorials, hybrid directions |

### Code

```
src/shogi_dojo/alphazero/
├── encoding.py     # Board → 45-channel tensor; Move ↔ 9×9×139 policy index
├── network.py      # ResNet (4 blocks × 64 filters) + policy/value heads (~336K params)
├── mcts.py         # Game-agnostic PUCT MCTS (select → expand → evaluate → backup)
├── rules.py        # python-shogi adapter: legal moves, terminal detection, network bridge
├── self_play.py    # play_game(): one full self-play game → training examples + replay buffer
└── trainer.py      # AlphaZeroTrainer: iteration loop (self-play → buffer → train → checkpoint)
```

### Quick Start (AlphaZero)

```bash
# Run the tiny smoke-test configuration (~minutes on CPU):
python train_alphazero.py --config configs/alphazero/tiny.yaml

# Run the default configuration (~hours on a single GPU):
python train_alphazero.py --config configs/alphazero/default.yaml
```

### Expectations

The survey's tiny default network (~340K params, 20 MCTS sims/move) will *not* approach competitive strength. It is designed to be read, run, and modified by the team as a learning tool. See [AlphaZero_Training_Reference.md §12](doc/alphazero/AlphaZero_Training_Reference.md#12-realistic-expectations-for-a-research-survey) for a frank discussion of what to expect.

## Bonanza-Method Survey

The third paradigm: supervised policy + value training from professional Shogi game records (kifu). Reuses the AlphaZero network architecture, with cross-entropy on the move played by the professional and MSE on the game outcome.

### Documentation

- [doc/bonanza/Bonanza_Reference.md](doc/bonanza/Bonanza_Reference.md) — history (Bonanza 2006 at WCSC26), the supervised classification formulation, data sources, three-paradigm comparison, strengths and limitations.

### Code

```
src/shogi_dojo/bonanza/
├── dataset.py    # CSA parser, game replay, PyTorch SupervisedDataset
└── trainer.py    # BonanzaTrainer: epoch-based supervised training loop
```

### Quick Start (Bonanza)

```bash
# Train on a CSA file (minutes on CPU with the tiny config):
python train_bonanza.py --data path/to/games.csa

# Use the default config with multiple files:
python train_bonanza.py --config configs/bonanza/default.yaml \
    --data file1.csa file2.csa --min-ply 10
```

You will need Shogi game records in CSA format. A good starting dataset is the [shogi-server kifu archive (2026-04-10 release)](https://github.com/shogi-server/shogi-server/releases/tag/2026-04-10) — a 7-Zip archive (`wdoor2025.7z`) containing ~128,843 CSA game files from the Floodgate server. See [Bonanza_Reference.md §5](doc/bonanza/Bonanza_Reference.md#5-data-professional-game-records) for extraction instructions, additional sources, and filtering guidance.

## Testing

```bash
# Python tests
pip install -e ".[dev]"
pytest tests/ -v
ruff check .

# Rust tests
cd tools/nnue/gensfen
cargo test
cargo clippy -- -D warnings
```

CI runs automatically on every push and PR via GitHub Actions.

## Contributing

Contributions are welcome. For routine changes, commit directly to `main`. For larger changes or when review is desired, use feature branches and pull requests.

## License

This project is provided as-is for research and educational purposes.
