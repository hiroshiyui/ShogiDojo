# TODO

## Rust gensfen Tool (`tools/gensfen/`)

Replace the YaneuraOu `gensfen` command with a standalone Rust tool that generates
NNUE training data (text SFEN format) without external engine dependencies.

### Why

YaneuraOu's `evallearn` build has compatibility issues with modern compilers
(GCC 15 runtime crash, V9.x `tt.h` regression). A self-contained Rust tool
eliminates this dependency entirely and outputs text SFEN directly (no binary
conversion step).

### Crate Dependencies

| Crate | Purpose |
|-------|---------|
| [shogi-rs](https://github.com/nozaq/shogi-rs) | Bitboard board representation, legal move generation, SFEN I/O |
| [haitaka](https://crates.io/crates/haitaka) | Alternative: faster move generation (evaluate if shogi-rs is too slow) |

### Implementation Tasks

- [x] Scaffold Rust project at `tools/gensfen/`
- [x] Board and move generation (via shogi-rs)
- [x] Alpha-beta search with iterative deepening
- [x] Phase 1 eval: material counting (piece values only)
- [x] Phase 2 eval: material + basic positional heuristics (piece-square tables)
- [x] Phase 3 eval: load trained NNUE weights from ShogiDojo checkpoints
- [x] Self-play game loop with opening randomization
- [x] Text SFEN output (position + score + game result per line)
- [x] Multi-threaded generation (rayon)
- [x] CLI interface (depth, thread count, number of positions, output path)

### Output Format

One position per line, directly consumable by ShogiDojo's `dataset.py`:

```
sfen <board> <side> <hand> <move#> <score> <result>
```

### Bootstrap Training Strategy

Training data quality depends on eval strength, but we can iterate:

1. **Phase 1** — Generate data with material-only eval. Train initial NNUE.
2. **Phase 2** — Add positional heuristics. Regenerate data. Retrain.
3. **Phase 3** — Load the trained NNUE into the Rust tool. Regenerate with
   the learned eval. Retrain. Repeat until convergence.

Each iteration produces stronger training data, which produces a stronger
NNUE, which produces even stronger training data.

## Other

- [ ] Write a `scripts/convert_bin_to_sfen.py` for converting YaneuraOu PackedSfenValue binary to text SFEN (still useful if someone has existing `.bin` data)
- [ ] Add pytest tests for feature extraction and model forward pass
- [ ] Configure ruff linting rules in `pyproject.toml`
