//! Shogi engine library for NNUE training data generation and model evaluation.
//!
//! This crate provides the core infrastructure for:
//!   - Position evaluation (PST and NNUE)
//!   - Alpha-beta search with quiescence
//!   - Pseudo-legal move generation
//!   - Self-play game loops
//!   - SFEN string generation
//!
//! Used by the `gensfen` binary (training data generation) and the
//! `eval_match` binary (model strength evaluation).

pub mod eval;
pub mod game;
pub mod match_game;
pub mod movegen;
pub mod nnue;
pub mod search;
pub mod sfen;
pub mod types;
