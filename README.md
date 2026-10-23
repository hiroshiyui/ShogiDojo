# Shogi Dojo — Computer Shogi Studies Research Group

Academic research resources for studying neural-network evaluation in Shogi (Japanese Chess), surveying three complementary paradigms side by side:

- **NNUE (HalfKAv2)** — a sparse, incrementally-updatable evaluation network (~5M params) designed for real-time alpha-beta search. The primary production pipeline: generate self-play data with a Rust tool, train with PyTorch, export quantized weights for engine integration.
- **AlphaZero-style** — a deep residual policy/value network (~336K params, survey-sized) trained purely from self-play via PUCT MCTS. An educational reference implementation; not a competitive engine.
- **Bonanza-method** — supervised policy + value training from professional Shogi game records (kifu). The simplest of the three (no MCTS, no self-play, just a dataset and a training loop). Reuses the AlphaZero network architecture and can warm-start AlphaZero self-play.

Each paradigm lives under its own subdirectory across `src/`, `configs/`, `scripts/`, `tests/`, and `doc/`. Top-level entry points are paradigm-prefixed: `train_nnue.py`, `train_alphazero.py`, `train_bonanza.py`.

## Documentation

| Paradigm | Reference | Training Methodology |
|---|---|---|
| NNUE       | [doc/nnue/NNUE_Reference.md](doc/nnue/NNUE_Reference.md) | [doc/nnue/Model_Training_Reference.md](doc/nnue/Model_Training_Reference.md) |
| AlphaZero  | [doc/alphazero/AlphaZero_Reference.md](doc/alphazero/AlphaZero_Reference.md) | [doc/alphazero/AlphaZero_Training_Reference.md](doc/alphazero/AlphaZero_Training_Reference.md) |
| Bonanza    | [doc/bonanza/Bonanza_Reference.md](doc/bonanza/Bonanza_Reference.md) | (covered in §4 of the reference) |

Cross-paradigm: [doc/alphazero/AlphaZero_vs_NNUE.md](doc/alphazero/AlphaZero_vs_NNUE.md) (operational comparison), [doc/alphazero/AlphaZero_Reading_List.md](doc/alphazero/AlphaZero_Reading_List.md) (annotated bibliography), [doc/TODO.md](doc/TODO.md) (roadmap).

## Project Structure

```
ShogiDojo/
├── train_nnue.py              # Entry point: NNUE training
├── train_alphazero.py         # Entry point: AlphaZero self-play training
├── train_bonanza.py           # Entry point: Bonanza supervised training
├── configs/{nnue,alphazero,bonanza}/   # Per-paradigm YAML configs
├── src/shogi_dojo/{nnue,alphazero,bonanza}/   # Implementations
├── scripts/{nnue,alphazero}/  # Helper scripts (export, evaluate, benchmark)
├── tests/{nnue,alphazero,bonanza}/    # Per-paradigm test suites
├── tools/nnue/gensfen/        # Rust: training-data generation + eval matches
├── doc/{nnue,alphazero,bonanza}/      # Per-paradigm reference documents
└── pyproject.toml
```

See [CLAUDE.md](CLAUDE.md) for the full module-by-module breakdown.

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.0
- NumPy, PyYAML
- python-shogi (used by the AlphaZero and Bonanza paradigms for legal-move generation and CSA parsing)
- Rust toolchain (for the NNUE `gensfen` and `eval_match` tools)

## Setup

```bash
git clone <repo-url>
cd ShogiDojo
source virtualenv/bin/activate

pip install -e .             # core dependencies
pip install -e ".[dev]"      # adds pytest + ruff
pip install -e ".[viz]"      # adds tensorboard
```

---

## Quick Start: NNUE

The end-to-end pipeline: build the Rust data generator → generate self-play positions → train the network → export weights.

```bash
# 1. Build the Rust tools
cd tools/nnue/gensfen && cargo build --release && cd ../../..

# 2. Generate training data (~100K positions via self-play with the PST evaluator)
mkdir -p data
./tools/nnue/gensfen/target/release/gensfen \
    --depth 4 --threads 4 --positions 100000 --output data/train.sfen

# 3. Train the NNUE model
python train_nnue.py
python train_nnue.py --config configs/nnue/small.yaml   # ~640K params, fast iteration
python train_nnue.py --config configs/nnue/large.yaml   # ~20M params, max strength
python train_nnue.py --resume checkpoints/epoch_0050.pt # continue from a checkpoint

# 4. Export the trained model
python scripts/nnue/export_nnue.py checkpoints/best_model.pt model.nnue
python scripts/nnue/export_nnue_quantized.py checkpoints/best_model.pt model_q.nnue
```

For stronger models, the iterative loop alternates data generation and training, each round bootstrapping the next:

```bash
python scripts/nnue/train_loop.py --iterations 3 --positions 100000
```

Evaluate model strength with paired color-swapped matches:

```bash
./tools/nnue/gensfen/target/release/eval_match \
    --model-a model_v2.nnue --model-b model_v1.nnue --games 200 --depth 4
```

For the full picture — `gensfen` flags and depth budget, SFEN format, dual-loss design, score scaling, ClippedReLU rationale, checkpoint cadence, improvement strategy, and disk budget — see [Model_Training_Reference.md](doc/nnue/Model_Training_Reference.md). Architecture deep-dive (HalfKAv2 features, king buckets, two-perspective computation) is in [NNUE_Reference.md](doc/nnue/NNUE_Reference.md).

---

## Quick Start: AlphaZero

A self-play loop driven by PUCT MCTS at every move. Reads, runs, and modifies are the goal — not competitive strength.

```bash
# Smoke test: a few iterations, runs in minutes on CPU
python train_alphazero.py --config configs/alphazero/tiny.yaml

# Laptop-scale training: hours on a single GPU
python train_alphazero.py --config configs/alphazero/default.yaml
```

Each iteration runs a batch of self-play games, appends the resulting (input, π, z) tuples to a replay buffer, and runs gradient steps on samples drawn from the buffer. Checkpoints are saved every N iterations.

The trainer accepts a `model.warm_start` config key pointing at any AlphaZero-architecture checkpoint — including a Bonanza-trained one (state_dicts are byte-for-byte interchangeable). This shortcuts the long bootstrapping phase where MCTS visits are uniformly random.

Compare two AlphaZero networks (or AlphaZero vs Bonanza) head-to-head:

```bash
python scripts/alphazero/match.py --network-a model_v2.pt --network-b model_v1.pt --num-games 100
```

Each network's size (blocks, filters, value-head width) is read from its own checkpoint, so networks of different sizes — e.g. the 10×128 Bonanza v2 against the 4×64 v1 — can play each other directly. MCTS runs in Python on one CPU core per process, so for large matches run several `match.py` processes in parallel (`--device cuda` lets them share the GPU) and add up their results.

For a sanity-floor opponent — useful for confirming a trained network actually plays better than random initialization — generate a freshly-initialized checkpoint with the same architecture:

```bash
python scripts/alphazero/make_random_checkpoint.py --output checkpoints/random_init.pt --seed 0
python scripts/alphazero/match.py \
    --network-a checkpoints/bonanza/bonanza_epoch_0050.pt \
    --network-b checkpoints/random_init.pt --num-games 20
```

A trained network should crush the random baseline ~95%+; if not, something is wrong (architecture mismatch, weight-loading bug, untrained checkpoint).

The survey's tiny default network will *not* approach competitive strength. See [AlphaZero_Training_Reference.md §12](doc/alphazero/AlphaZero_Training_Reference.md#12-realistic-expectations-for-a-research-survey) for an honest discussion of expectations, and [AlphaZero_Reference.md](doc/alphazero/AlphaZero_Reference.md) for architecture, MCTS+PUCT walk-through, and the 9×9×139 action-space encoding.

---

## Quick Start: Bonanza

Supervised learning on professional kifu — no MCTS, no self-play, just a dataset and a standard PyTorch training loop. Cross-entropy on the policy head against the move played by the professional, MSE on the value head against the game outcome from this position's side-to-move.

```bash
# Smoke test on a single CSA file (minutes on CPU with the tiny config):
python train_bonanza.py --data path/to/games.csa

# Default config with multiple files, skipping the first 10 plies (book moves):
python train_bonanza.py --config configs/bonanza/default.yaml \
    --data file1.csa file2.csa --min-ply 10

# Large dataset (~167K wdoor files): use --data-list (avoids argv limit)
# and --streaming (bounded-memory IterableDataset):
find data/wdoor -name '*.csa' > data/wdoor_files.txt
python train_bonanza.py --config configs/bonanza/default.yaml \
    --data-list data/wdoor_files.txt --streaming --min-ply 10

# Resume from a saved checkpoint (epoch- or step-based):
python train_bonanza.py --resume checkpoints/bonanza_epoch_0010.pt \
    --data-list data/wdoor_files.txt --streaming
```

### Deploying the Trained Network

A trained checkpoint is a standard PyTorch `.pt` file (model weights + optimizer state). For inference outside PyTorch — mobile apps, browsers, C++ engines — convert it to a portable format:

```bash
# Install export extras (onnx + onnxruntime)
pip install -e ".[export]"

# ONNX (recommended for mobile/web — Core ML, TFLite, ONNX Runtime Web all consume it):
python scripts/alphazero/export_onnx.py \
    --checkpoint checkpoints/bonanza/bonanza_epoch_0050.pt \
    --output bonanza.onnx

# TorchScript (when the deployment target already uses libtorch C++):
python scripts/alphazero/export_torchscript.py \
    --checkpoint checkpoints/bonanza/bonanza_epoch_0050.pt \
    --output bonanza.ts
```

For the v1 4×64 network both produce ~1.3 MB artifacts (336K params at float32) with named I/O and a dynamic batch dimension, so the same file works for one-position-at-a-time mobile inference and for batched server inference.

Two exported networks are committed at the repository root, with identical I/O:

| File | Network | Params | Size | Notes |
|---|---|---|---|---|
| `bonanza.onnx` | v1, 4×64 | 336K | 1.3 MB | 44.5% move accuracy |
| `bonanza_v2.onnx` | v2, 10×128 | 3.0M | 12 MB | 54.3% held-out accuracy; beat v1 208–0 |

Larger networks must be exported with matching size flags, e.g. `--num-blocks 10 --num-filters 128 --value-head-hidden 128` for v2.

The exported file contains only the network — to actually use it in a deployment language (Swift, Kotlin, JavaScript, Rust) you also need an encoder, a legal-move mask, a policy decoder, and the value sign-flip convention. [doc/bonanza/Deployment.md](doc/bonanza/Deployment.md) is the single-source-of-truth contract for those pieces.

### Why Bonanza?

- **Simplest pipeline.** No replay buffer, no Dirichlet noise, no virtual loss — a few thousand lines including tests. Good first paradigm to read.
- **Strong baseline cheaply.** A few hours on commodity hardware over the wdoor archive produces a network that already plays sensible Shogi, because professionals are excellent demonstrators.
- **AlphaZero warm-start.** Because the network architecture is identical, a Bonanza checkpoint loads directly into `AlphaZeroTrainer` via the `model.warm_start` config key. Self-play then begins from a network that already knows opening theory and standard middlegame ideas, skipping the long random-policy bootstrap.

### Dataset Modes

| Mode | Class | When to use | Memory |
|---|---|---|---|
| Map-style  | `SupervisedDataset`           | Up to ~a few thousand games / a few GB of samples | Loads everything at construction |
| Streaming  | `StreamingSupervisedDataset`  | Large archives (the full wdoor would be ~200 GB if loaded) | Bounded by `--buffer-size` (default 10K samples) |

`--streaming` switches to the streaming dataset; both produce identically-shaped batches.

### Checkpoints

Two independent triggers configurable in the YAML:

- `save_every` (epochs) — periodic snapshot, default `10`.
- `save_every_steps` (optimizer steps) — mid-epoch snapshot. Essential for the wdoor archive where one epoch can take hours; without it a crash loses everything.

Resume from either kind via `--resume PATH`. Step-checkpoint resumes restart cleanly at the next epoch (mid-epoch resume would require persisting DataLoader/RNG state).

### Data: Where to Get Kifu

The recommended starting dataset is the [shogi-server kifu archive (2026-04-10 release)](https://github.com/shogi-server/shogi-server/releases/tag/2026-04-10) — `wdoor2025.7z` + `wdoor2026.7z`, ~166,740 CSA games combined from the Floodgate server. See [Bonanza_Reference.md §5](doc/bonanza/Bonanza_Reference.md#5-data-professional-game-records) for extraction instructions, additional sources (Shogi24, 81Dojo dumps), CSA format details, and quality filtering guidance (rating thresholds, decisive-result filter).

For the Bonanza method's history (Bonanza 2006 at WCSC26), the formulation derivation, GPU utilization tuning, and a frank three-paradigm comparison, see [Bonanza_Reference.md](doc/bonanza/Bonanza_Reference.md).

---

## Testing

```bash
# Python tests across all paradigms
pip install -e ".[dev]"
pytest tests/
ruff check .

# Rust tests for the NNUE tools
cd tools/nnue/gensfen
cargo test
cargo clippy -- -D warnings
```

CI runs both suites on every push/PR via GitHub Actions (`.github/workflows/ci.yml`).

## Contributing

Routine changes go directly to `main`. For larger work or when review is desired, use a feature branch and open a pull request. Every code addition or change should come with corresponding tests — see the testing policy in [CLAUDE.md](CLAUDE.md).

## License

Provided as-is for research and educational purposes.
