# TODO

## Rust gensfen Tool (`tools/gensfen/`) — Done

Standalone Rust tool replacing YaneuraOu for NNUE training data generation.
Features: alpha-beta search with iterative deepening, three-phase eval
(material → positional → NNUE), self-play with opening randomization,
multi-threaded generation (rayon), and text SFEN output directly consumable
by `dataset.py`.

## Other

- [ ] Write a `scripts/convert_bin_to_sfen.py` for converting YaneuraOu PackedSfenValue binary to text SFEN (still useful if someone has existing `.bin` data)
- [ ] Add pytest tests for feature extraction and model forward pass
- [ ] Configure ruff linting rules in `pyproject.toml`
