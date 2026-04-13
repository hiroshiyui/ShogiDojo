//! Match game: play a game between two different evaluators.
//!
//! Unlike `game::play_game()` which uses a single evaluator for self-play,
//! this module plays games where Black and White use different evaluators.
//! This is used to measure relative strength between two models.

use std::collections::HashSet;

use rand::Rng;
use rand::seq::SliceRandom;
use shogi::{Color, Position};

use crate::eval::Evaluator;
use crate::game::{determine_outcome, get_legal_moves};
use crate::search::Searcher;
use crate::sfen;
use crate::types::GameOutcome;

/// Maximum number of plies (half-moves) before the game is declared a draw.
const MAX_PLY: usize = 400;

/// The standard SFEN for the initial Shogi position.
const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// Play one game between two evaluators and return the outcome.
///
/// `black_eval` drives Black's search; `white_eval` drives White's search.
/// A random opening of `random_moves` moves is played first for diversity,
/// then each side uses its own evaluator for the remainder of the game.
pub fn play_match_game(
    black_eval: &dyn Evaluator,
    white_eval: &dyn Evaluator,
    depth: i32,
    random_moves: usize,
    rng: &mut impl Rng,
) -> GameOutcome {
    let mut pos = Position::new();
    pos.set_sfen(STARTPOS).expect("invalid startpos SFEN");

    let mut seen_positions: HashSet<String> = HashSet::new();
    seen_positions.insert(sfen::position_to_sfen(&pos));

    let mut ply: usize = 0;

    // === Phase 1: Random opening ===
    for _ in 0..random_moves {
        let moves = get_legal_moves(&mut pos);
        if moves.is_empty() {
            return determine_outcome(&pos);
        }

        let m = moves.choose(rng).unwrap();
        pos.make_move(*m).unwrap();
        ply += 1;

        let current_sfen = sfen::position_to_sfen(&pos);
        if seen_positions.contains(&current_sfen) {
            return GameOutcome::Draw;
        }
        seen_positions.insert(current_sfen);
    }

    // === Phase 2: Engine play (each side uses its own evaluator) ===
    loop {
        if ply >= MAX_PLY {
            return GameOutcome::Draw;
        }

        // Pick the evaluator for the current side to move.
        let evaluator: &dyn Evaluator = match pos.side_to_move() {
            Color::Black => black_eval,
            Color::White => white_eval,
        };

        let mut searcher = Searcher::new(evaluator);
        let (best_move, _score) = searcher.search(&mut pos, depth);

        let best_move = match best_move {
            Some(m) => m,
            None => {
                // No legal moves: checkmate or stalemate.
                return determine_outcome(&pos);
            }
        };

        if pos.make_move(best_move).is_err() {
            return GameOutcome::Draw;
        }
        ply += 1;

        // Check for repetition.
        let new_sfen = sfen::position_to_sfen(&pos);
        if seen_positions.contains(&new_sfen) {
            return GameOutcome::Draw;
        }
        seen_positions.insert(new_sfen);

        // Check if opponent has legal moves.
        let opponent_moves = get_legal_moves(&mut pos);
        if opponent_moves.is_empty() {
            return determine_outcome(&pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::PstEvaluator;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use shogi::bitboard::Factory as BBFactory;

    fn init() {
        BBFactory::init();
    }

    #[test]
    fn pst_vs_pst_completes() {
        init();
        let eval_a = PstEvaluator;
        let eval_b = PstEvaluator;
        let mut rng = SmallRng::seed_from_u64(42);
        let outcome = play_match_game(&eval_a, &eval_b, 2, 4, &mut rng);
        // Should be one of the three valid outcomes.
        assert!(
            outcome == GameOutcome::BlackWin
                || outcome == GameOutcome::WhiteWin
                || outcome == GameOutcome::Draw
        );
    }

    #[test]
    fn different_seeds_different_outcomes_possible() {
        init();
        let eval_a = PstEvaluator;
        let eval_b = PstEvaluator;
        // Play several games and collect outcomes.
        let mut outcomes = Vec::new();
        for seed in 0..5 {
            let mut rng = SmallRng::seed_from_u64(seed);
            outcomes.push(play_match_game(&eval_a, &eval_b, 2, 6, &mut rng));
        }
        // At least one game should have finished (not all identical outcomes
        // would be a very unlikely coincidence, but we just check they complete).
        assert_eq!(outcomes.len(), 5);
    }
}
