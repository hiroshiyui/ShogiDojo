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

use crate::eval;
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
pub struct Searcher {
    /// Number of positions (nodes) evaluated during the current search.
    pub nodes: u64,
}

impl Searcher {
    pub fn new() -> Self {
        Self { nodes: 0 }
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
            // Try to make the move. If it's illegal (e.g., leaves king in
            // check, nifu, uchifuzume), make_move returns an error and we skip.
            if pos.make_move(*m).is_err() {
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
                // This can happen in rare edge cases with the library.
                return best_score.max(alpha);
            }

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
        let stand_pat = eval::evaluate(pos);

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
        let captures: Vec<Move> = moves
            .into_iter()
            .filter(|m| is_capture(pos, m))
            .collect();

        let scored = order_moves(pos, &captures);

        for (_, m) in &scored {
            if pos.make_move(*m).is_err() {
                continue;
            }

            let score = -self.quiescence(pos, -beta, -alpha);

            if pos.unmake_move().is_err() {
                return alpha;
            }

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

    // Sort descending by score (best moves first)
    scored.sort_by(|a, b| b.0.cmp(&a.0));
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
