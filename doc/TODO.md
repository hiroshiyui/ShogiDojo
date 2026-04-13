# TODO

## Rust gensfen Tool (`tools/gensfen/`) — Done

Standalone Rust tool replacing YaneuraOu for NNUE training data generation.
Features: alpha-beta search with iterative deepening, three-phase eval
(material → positional → NNUE), self-play with opening randomization,
multi-threaded generation (rayon), and text SFEN output directly consumable
by `dataset.py`.

