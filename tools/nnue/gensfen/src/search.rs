//! Alpha-beta search with quiescence for Shogi.
//!
//! The search finds the best move in a position by exploring the game tree
//! using alpha-beta pruning — a technique that skips branches of the tree
//! that provably cannot affect the result.
//!
//! Key concepts:
//!   - **Alpha**: the minimum score the side-to-move is guaranteed.
//!   - **Beta**: the maximum score the opponent will allow.
//!   - **Pruning**: if we find a move that scores >= beta, we can stop
//!     searching this node (the opponent would never allow this position).
//!   - **Quiescence search**: at the search depth limit, we continue
//!     searching capture moves to avoid evaluating positions in the middle
//!     of a piece exchange (the "horizon effect").
//!
//! This is a simple implementation without advanced techniques like
//! null-move pruning, late-move reductions, or killer moves. Those can be
//! added in later phases for better performance.

use shogi::{Move, PieceType, Position};

use crate::eval::Evaluator;
use crate::movegen;

/// Score representing a won game (checkmate). The depth offset ensures
/// that a faster checkmate scores higher than a slower one.
pub const MATE_SCORE: i32 = 90_000;

/// Upper bound for scores. Used as the initial alpha/beta window.
pub const INFINITY: i32 = 100_000;

/// Search state, owned by a single thread.
///
/// Each thread in the rayon pool creates its own Searcher. There is no
/// shared state between searchers — this avoids synchronization overhead.
///
/// The evaluator is borrowed mutably so that stateful evaluators (NNUE)
/// can keep an incremental accumulator in sync with make/unmake.
pub struct Searcher<'a> {
    /// Number of positions (nodes) evaluated during the current search.
    pub nodes: u64,
    /// The evaluation function to use (PST or NNUE).
    evaluator: &'a mut dyn Evaluator,
}

impl<'a> Searcher<'a> {
    pub fn new(evaluator: &'a mut dyn Evaluator) -> Self {
        Self {
            nodes: 0,
            evaluator,
        }
    }

    /// Find the best move in a position using iterative deepening.
    ///
    /// Iterative deepening searches at depth 1, then depth 2, etc., up to
    /// `max_depth`. Each iteration reuses knowledge from the previous one
    /// (via better move ordering). This also gives us a best move even if
    /// we had to stop early.
    ///
    /// Returns `(best_move, score)`. Returns `None` for best_move if no
    /// legal moves exist (checkmate or stalemate).
    pub fn search(&mut self, pos: &mut Position, max_depth: i32) -> (Option<Move>, i32) {
        self.nodes = 0;
        // Resync the evaluator's accumulator with the current root position.
        // Stateless evaluators (PST) ignore this.
        self.evaluator.refresh(pos);
        let mut best_move = None;
        let mut best_score = -INFINITY;

        // Iterative deepening: search at increasing depths
        for depth in 1..=max_depth {
            let score = self.alpha_beta(pos, depth, -INFINITY, INFINITY, &mut best_move);
            best_score = score;
        }

        (best_move, best_score)
    }

    /// Alpha-beta negamax search.
    ///
    /// "Negamax" is a simplification of minimax where we always evaluate
    /// from the side-to-move's perspective and negate the score when
    /// recursing. This avoids separate "maximize" and "minimize" branches.
    ///
    /// Returns the evaluation score from the current side-to-move's perspective.
    fn alpha_beta(
        &mut self,
        pos: &mut Position,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        root_best: &mut Option<Move>,
    ) -> i32 {
        // At depth 0, switch to quiescence search to resolve captures.
        if depth <= 0 {
            return self.quiescence(pos, alpha, beta);
        }

        let moves = movegen::generate_moves(pos);
        // Sort moves: captures first (better move ordering = more pruning).
        let scored_moves = order_moves(pos, &moves);

        let mut legal_move_count = 0;
        let mut best_score = -INFINITY;

        for (_, m) in &scored_moves {
            // Snapshot the evaluator state and pre-compute the forward
            // accumulator delta, *before* make_move. Stateless evaluators
            // (PST) ignore this.
            self.evaluator.on_make(pos, m);

            // Try to make the move. If it's illegal (e.g., leaves king in
            // check, nifu, uchifuzume), make_move returns an error and we skip.
            if pos.make_move(*m).is_err() {
                // Roll back the speculative on_make.
                self.evaluator.on_unmake();
                continue;
            }
            legal_move_count += 1;

            // Recurse: search the resulting position from the opponent's view.
            // The negation implements the "negamax" trick.
            let mut dummy = None;
            let score = -self.alpha_beta(pos, depth - 1, -beta, -alpha, &mut dummy);

            // Undo the move to restore the position for the next candidate.
            if pos.unmake_move().is_err() {
                // shogi-rs hit an internal inconsistency — abort this node.
                // Force a full evaluator resync since we can no longer trust
                // the snapshot stack to match the position's true history.
                self.evaluator.refresh(pos);
                return best_score.max(alpha);
            }
            self.evaluator.on_unmake();

            if score > best_score {
                best_score = score;
                // Update root best move (only at the top level of each ID iteration)
                if depth >= 1 {
                    *root_best = Some(*m);
                }
            }

            // Alpha-beta pruning:
            // Alpha is our "guaranteed minimum" — if we find something better, raise it.
            if score > alpha {
                alpha = score;
            }
            // Beta cutoff: the opponent would never allow this position,
            // so we can stop searching the remaining moves.
            if alpha >= beta {
                break;
            }
        }

        // If no legal moves were found, it's either checkmate or stalemate.
        if legal_move_count == 0 {
            if pos.in_check(pos.side_to_move()) {
                // Checkmate: the worst possible score. Subtract depth so that
                // a faster checkmate is preferred (scores closer to -MATE_SCORE).
                return -MATE_SCORE + depth;
            } else {
                // Stalemate: in Shogi, this is extremely rare. The player with
                // no legal moves loses. Treat as a loss.
                return -MATE_SCORE + depth;
            }
        }

        best_score
    }

    /// Quiescence search: continue searching capture moves at the leaves.
    ///
    /// Without quiescence, the search might evaluate a position right after
    /// a piece is captured but before the recapture — giving a wildly
    /// inaccurate score. Quiescence search resolves these "noisy" positions
    /// by searching all captures until the position is "quiet".
    ///
    /// The "stand-pat" score is the static evaluation of the current position.
    /// If this is already >= beta, we prune (the position is already too good
    /// for us — the opponent wouldn't have allowed it).
    fn quiescence(&mut self, pos: &mut Position, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;

        // Stand-pat: evaluate the position without making any more moves.
        let stand_pat = self.evaluator.evaluate(pos);

        // Beta cutoff: position is already too good for us.
        if stand_pat >= beta {
            return beta;
        }

        // If stand-pat improves alpha, update it.
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        // Generate only capture moves (normal moves to occupied squares).
        let moves = movegen::generate_moves(pos);
        let captures: Vec<Move> = moves.into_iter().filter(|m| is_capture(pos, m)).collect();

        let scored = order_moves(pos, &captures);

        for (_, m) in &scored {
            self.evaluator.on_make(pos, m);
            if pos.make_move(*m).is_err() {
                self.evaluator.on_unmake();
                continue;
            }

            let score = -self.quiescence(pos, -beta, -alpha);

            if pos.unmake_move().is_err() {
                self.evaluator.refresh(pos);
                return alpha;
            }
            self.evaluator.on_unmake();

            if score >= beta {
                return beta; // Beta cutoff
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }
}

/// Check if a move is a capture (the destination square has an opponent's piece).
fn is_capture(pos: &Position, m: &Move) -> bool {
    match m {
        Move::Normal { to, .. } => pos.piece_at(*to).is_some(),
        Move::Drop { .. } => false, // Drops are never captures
    }
}

/// Order moves for better alpha-beta pruning.
///
/// Good move ordering is critical for search efficiency. With perfect
/// ordering, alpha-beta searches O(b^(d/2)) nodes instead of O(b^d).
///
/// We use MVV-LVA (Most Valuable Victim - Least Valuable Attacker):
/// captures of high-value pieces by low-value pieces are searched first.
/// Non-captures get a score of 0 (searched last).
fn order_moves(pos: &Position, moves: &[Move]) -> Vec<(i32, Move)> {
    let mut scored: Vec<(i32, Move)> = moves
        .iter()
        .map(|m| {
            let score = match m {
                Move::Normal { from, to, promote } => {
                    let mut s = 0i32;
                    // Captures: prioritize capturing valuable pieces with cheap pieces
                    if let Some(victim) = pos.piece_at(*to) {
                        let victim_val = piece_type_value(victim.piece_type);
                        let attacker_val = pos
                            .piece_at(*from)
                            .map(|p| piece_type_value(p.piece_type))
                            .unwrap_or(0);
                        s += victim_val * 10 - attacker_val; // MVV-LVA
                    }
                    // Promotions are usually good
                    if *promote {
                        s += 500;
                    }
                    s
                }
                Move::Drop { .. } => 0, // Drops are searched after captures
            };
            (score, *m)
        })
        .collect();

    // Sort descending by score (best moves first). Using sort_by_key
    // with std::cmp::Reverse satisfies clippy::unnecessary_sort_by
    // introduced in Rust 1.95.
    scored.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    scored
}

/// Simplified piece values for move ordering (MVV-LVA).
fn piece_type_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn | PieceType::ProPawn => 1,
        PieceType::Lance | PieceType::ProLance => 3,
        PieceType::Knight | PieceType::ProKnight => 4,
        PieceType::Silver | PieceType::ProSilver => 5,
        PieceType::Gold => 5,
        PieceType::Bishop | PieceType::ProBishop => 7,
        PieceType::Rook | PieceType::ProRook => 9,
        PieceType::King => 100, // Capturing king should never happen, but rank it high
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::PstEvaluator;
    use shogi::Square;
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
    // piece_type_value — ordering sanity
    // -----------------------------------------------------------------------

    #[test]
    fn piece_ordering_hierarchy() {
        assert!(piece_type_value(PieceType::Rook) > piece_type_value(PieceType::Bishop));
        assert!(piece_type_value(PieceType::Bishop) > piece_type_value(PieceType::Gold));
        assert!(piece_type_value(PieceType::Gold) >= piece_type_value(PieceType::Silver));
        assert!(piece_type_value(PieceType::Silver) > piece_type_value(PieceType::Knight));
        assert!(piece_type_value(PieceType::Knight) > piece_type_value(PieceType::Lance));
        assert!(piece_type_value(PieceType::Lance) > piece_type_value(PieceType::Pawn));
        assert!(piece_type_value(PieceType::King) > piece_type_value(PieceType::Rook));
    }

    #[test]
    fn promoted_pieces_equal_to_base() {
        assert_eq!(
            piece_type_value(PieceType::Pawn),
            piece_type_value(PieceType::ProPawn)
        );
        assert_eq!(
            piece_type_value(PieceType::Lance),
            piece_type_value(PieceType::ProLance)
        );
        assert_eq!(
            piece_type_value(PieceType::Bishop),
            piece_type_value(PieceType::ProBishop)
        );
        assert_eq!(
            piece_type_value(PieceType::Rook),
            piece_type_value(PieceType::ProRook)
        );
    }

    // -----------------------------------------------------------------------
    // is_capture
    // -----------------------------------------------------------------------

    #[test]
    fn drop_is_not_capture() {
        let pos = make_pos("4k4/9/9/9/9/9/9/9/4K4 b P 1");
        let m = Move::Drop {
            to: Square::new(4, 4).unwrap(),
            piece_type: PieceType::Pawn,
        };
        assert!(!is_capture(&pos, &m));
    }

    #[test]
    fn move_to_empty_is_not_capture() {
        let pos = make_pos("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        let m = Move::Normal {
            from: Square::new(4, 8).unwrap(), // King at e1
            to: Square::new(3, 8).unwrap(),   // Empty square
            promote: false,
        };
        assert!(!is_capture(&pos, &m));
    }

    #[test]
    fn move_to_occupied_is_capture() {
        // Black King on 5i, White Pawn on 4i.
        let pos = make_pos("4k4/9/9/9/9/9/9/9/3pK4 b - 1");
        let m = Move::Normal {
            from: Square::new(4, 8).unwrap(), // King at file 4, rank 8
            to: Square::new(5, 8).unwrap(),   // White pawn at file 5, rank 8
            promote: false,
        };
        assert!(is_capture(&pos, &m));
    }

    // -----------------------------------------------------------------------
    // Searcher — integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn search_returns_a_move_from_startpos() {
        let mut evaluator = PstEvaluator;
        let mut searcher = Searcher::new(&mut evaluator);
        let mut pos = make_pos("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
        let (best_move, _score) = searcher.search(&mut pos, 2);
        assert!(
            best_move.is_some(),
            "search should find a move from startpos"
        );
    }

    #[test]
    fn search_nodes_are_counted() {
        let mut evaluator = PstEvaluator;
        let mut searcher = Searcher::new(&mut evaluator);
        let mut pos = make_pos("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
        searcher.search(&mut pos, 2);
        assert!(
            searcher.nodes > 0,
            "search should evaluate at least one node"
        );
    }

    #[test]
    fn search_depth_1_returns_valid_move() {
        let mut evaluator = PstEvaluator;
        let mut searcher = Searcher::new(&mut evaluator);
        let mut pos = make_pos("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        let (best_move, _) = searcher.search(&mut pos, 1);
        assert!(best_move.is_some());
    }

    #[test]
    fn search_score_is_finite() {
        let mut evaluator = PstEvaluator;
        let mut searcher = Searcher::new(&mut evaluator);
        let mut pos = make_pos("4k4/9/9/9/9/9/9/9/4K4 b - 1");
        let (_, score) = searcher.search(&mut pos, 2);
        assert!(
            score.abs() < INFINITY,
            "score should be finite, got {score}"
        );
    }
}
