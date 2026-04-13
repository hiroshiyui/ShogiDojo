//! Self-play game loop for generating training data.
//!
//! A single "game" works like this:
//!
//!   1. Start from the standard initial Shogi position.
//!   2. Play `random_moves` random moves to create a diverse opening.
//!      (Without this, every game would start identically and the training
//!      data would be heavily biased toward the starting position.)
//!   3. Switch to engine play: at each position, run an alpha-beta search
//!      to find the best move and its evaluation score.
//!   4. Record each position's SFEN and search score.
//!   5. Continue until the game ends (checkmate, max ply, or repetition).
//!   6. After the game ends, determine the outcome (who won?) and "backfill"
//!      the result into all recorded positions.
//!   7. Return the finished training samples.

use std::collections::HashSet;

use rand::Rng;
use rand::seq::SliceRandom;
use shogi::{Color, Position};

use crate::eval::Evaluator;
use crate::movegen;
use crate::search::Searcher;
use crate::sfen;
use crate::types::{GameOutcome, RecordedPosition, TrainingSample};

/// Maximum number of plies (half-moves) before the game is declared a draw.
/// This prevents infinite games in positions where neither side can make progress.
const MAX_PLY: usize = 400;

/// The standard SFEN for the initial Shogi position.
const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// Play one complete self-play game and return training samples.
///
/// Each sample is a position from the game with:
///   - The SFEN string
///   - The search evaluation (from the side-to-move's perspective)
///   - The actual game result (from the side-to-move's perspective)
///
/// Returns an empty Vec if the game couldn't produce useful data (e.g.,
/// the random opening phase resulted in an immediate end).
pub fn play_game(
    depth: i32,
    random_moves: usize,
    rng: &mut impl Rng,
    evaluator: &dyn Evaluator,
) -> Vec<TrainingSample> {
    let mut pos = Position::new();
    pos.set_sfen(STARTPOS).expect("invalid startpos SFEN");

    let mut searcher = Searcher::new(evaluator);
    let mut recorded: Vec<RecordedPosition> = Vec::new();

    // Track seen positions for repetition detection.
    // We use our own tracking because shogi-rs's repetition detection
    // can corrupt position state on error (known bug).
    let mut seen_positions: HashSet<String> = HashSet::new();
    seen_positions.insert(sfen::position_to_sfen(&pos));

    let mut ply: usize = 0;
    let mut outcome: Option<GameOutcome> = None;

    // === Phase 1: Random opening ===
    // Play random legal moves to create diverse starting positions.
    // This is critical for training data quality — without diversity,
    // the NNUE would only learn about positions reachable from one
    // specific opening sequence.
    for _ in 0..random_moves {
        let moves = get_legal_moves(&mut pos);
        if moves.is_empty() {
            // Game ended during random opening — unusual but possible.
            outcome = Some(determine_outcome(&pos));
            break;
        }

        // Pick a random legal move
        let m = moves.choose(rng).unwrap();
        pos.make_move(*m).unwrap();
        ply += 1;

        let current_sfen = sfen::position_to_sfen(&pos);
        if seen_positions.contains(&current_sfen) {
            outcome = Some(GameOutcome::Draw); // Repetition → draw
            break;
        }
        seen_positions.insert(current_sfen);
    }

    // === Phase 2: Engine play ===
    // Use alpha-beta search to play the rest of the game.
    // Record each position for training data.
    while outcome.is_none() && ply < MAX_PLY {
        let sfen = sfen::position_to_sfen(&pos);

        // Search for the best move
        let (best_move, score) = searcher.search(&mut pos, depth);

        let best_move = match best_move {
            Some(m) => m,
            None => {
                // No legal moves: checkmate or stalemate
                outcome = Some(determine_outcome(&pos));
                // Still record this terminal position
                recorded.push(RecordedPosition {
                    sfen,
                    score,
                    side_to_move: pos.side_to_move(),
                });
                break;
            }
        };

        // Record this position with its search score
        recorded.push(RecordedPosition {
            sfen,
            score,
            side_to_move: pos.side_to_move(),
        });

        // Make the best move
        if pos.make_move(best_move).is_err() {
            // Shouldn't happen (search validated the move), but handle gracefully
            outcome = Some(GameOutcome::Draw);
            break;
        }
        ply += 1;

        // Check for repetition (using our own tracking)
        let new_sfen = sfen::position_to_sfen(&pos);
        if seen_positions.contains(&new_sfen) {
            outcome = Some(GameOutcome::Draw);
            break;
        }
        seen_positions.insert(new_sfen);

        // Check for checkmate after the move (opponent has no legal moves)
        let opponent_moves = get_legal_moves(&mut pos);
        if opponent_moves.is_empty() {
            outcome = Some(determine_outcome(&pos));
        }
    }

    // If we hit the ply limit without a decisive result, it's a draw.
    let outcome = outcome.unwrap_or(GameOutcome::Draw);

    // === Phase 3: Backfill results ===
    // Now that we know the game outcome, convert all recorded positions
    // into training samples with the correct result annotation.
    recorded
        .into_iter()
        .map(|rec| rec.resolve(outcome))
        .collect()
}

/// Get all legal moves in the current position.
///
/// Generates pseudo-legal moves, then filters by actually trying each one.
/// Moves that fail `make_move()` (leave king in check, nifu, etc.) are excluded.
fn get_legal_moves(pos: &mut Position) -> Vec<shogi::Move> {
    let pseudo_legal = movegen::generate_moves(pos);
    let mut legal = Vec::new();

    for m in pseudo_legal {
        if pos.make_move(m).is_ok() {
            if pos.unmake_move().is_err() {
                // shogi-rs internal inconsistency — skip this move
                continue;
            }
            legal.push(m);
        }
    }

    legal
}

/// Determine the game outcome when the side-to-move has no legal moves.
///
/// In Shogi:
///   - If the side-to-move is in check with no legal moves → checkmate (they lose)
///   - If the side-to-move is NOT in check with no legal moves → stalemate (they lose)
///     (Unlike chess where stalemate is a draw, in Shogi the stalemated player loses.
///     However, stalemate is extremely rare in practice.)
fn determine_outcome(pos: &Position) -> GameOutcome {
    // The side-to-move has no legal moves → they lose.
    match pos.side_to_move() {
        Color::Black => GameOutcome::WhiteWin,
        Color::White => GameOutcome::BlackWin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use shogi::bitboard::Factory as BBFactory;

    fn init() {
        BBFactory::init();
    }

    fn make_pos(sfen: &str) -> Position {
        init();
        let mut pos = Position::new();
        pos.set_sfen(sfen).unwrap();
        pos
    }

    // -----------------------------------------------------------------------
    // determine_outcome
    // -----------------------------------------------------------------------

    #[test]
    fn black_to_move_no_moves_white_wins() {
        let pos = make_pos("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        assert_eq!(determine_outcome(&pos), GameOutcome::WhiteWin);
    }

    #[test]
    fn white_to_move_no_moves_black_wins() {
        let pos = make_pos("4k4/9/9/9/9/9/9/9/4K4 w - 1");
        assert_eq!(determine_outcome(&pos), GameOutcome::BlackWin);
    }

    // -----------------------------------------------------------------------
    // get_legal_moves
    // -----------------------------------------------------------------------

    #[test]
    fn startpos_has_30_legal_moves() {
        let mut pos = make_pos(STARTPOS);
        let moves = get_legal_moves(&mut pos);
        assert_eq!(
            moves.len(),
            30,
            "startpos should have 30 legal moves, got {}",
            moves.len()
        );
    }

    #[test]
    fn bare_kings_have_limited_moves() {
        let mut pos = make_pos("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        let moves = get_legal_moves(&mut pos);
        // King in center of back rank has 5 moves (3 forward, 2 diagonal).
        assert!(
            moves.len() <= 5,
            "bare king should have <= 5 moves, got {}",
            moves.len()
        );
        assert!(!moves.is_empty());
    }

    // -----------------------------------------------------------------------
    // play_game — smoke test
    // -----------------------------------------------------------------------

    #[test]
    fn play_game_produces_samples() {
        init();
        let evaluator = crate::eval::PstEvaluator;
        let mut rng = SmallRng::seed_from_u64(42);
        let samples = play_game(2, 4, &mut rng, &evaluator);
        assert!(
            !samples.is_empty(),
            "play_game should produce at least one sample"
        );
    }

    #[test]
    fn play_game_samples_have_valid_results() {
        init();
        let evaluator = crate::eval::PstEvaluator;
        let mut rng = SmallRng::seed_from_u64(123);
        let samples = play_game(2, 4, &mut rng, &evaluator);
        for sample in &samples {
            assert!(
                sample.result == 0.0 || sample.result == 0.5 || sample.result == 1.0,
                "result should be 0.0, 0.5, or 1.0, got {}",
                sample.result
            );
            assert!(sample.sfen.len() > 10, "SFEN too short: {}", sample.sfen);
        }
    }

    #[test]
    fn play_game_different_seeds_produce_different_games() {
        init();
        let evaluator = crate::eval::PstEvaluator;
        let mut rng1 = SmallRng::seed_from_u64(1);
        let mut rng2 = SmallRng::seed_from_u64(999);
        let samples1 = play_game(2, 6, &mut rng1, &evaluator);
        let samples2 = play_game(2, 6, &mut rng2, &evaluator);
        // Different seeds → (very likely) different games.
        // Check first non-trivial position differs.
        if !samples1.is_empty() && !samples2.is_empty() {
            let sfens1: Vec<&str> = samples1.iter().map(|s| s.sfen.as_str()).collect();
            let sfens2: Vec<&str> = samples2.iter().map(|s| s.sfen.as_str()).collect();
            assert_ne!(
                sfens1, sfens2,
                "different seeds should produce different games"
            );
        }
    }

    #[test]
    fn all_samples_have_sfen_prefix_in_to_line() {
        init();
        let evaluator = crate::eval::PstEvaluator;
        let mut rng = SmallRng::seed_from_u64(777);
        let samples = play_game(2, 4, &mut rng, &evaluator);
        for sample in &samples {
            let line = sample.to_line();
            assert!(
                line.starts_with("sfen "),
                "line should start with 'sfen ': {line}"
            );
            // Verify tab-separated format is parseable.
            let parts: Vec<&str> = line.split('\t').collect();
            assert_eq!(parts.len(), 3, "expected 3 tab-separated fields: {line}");
        }
    }
}
