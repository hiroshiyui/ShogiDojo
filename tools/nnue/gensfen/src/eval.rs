//! Position evaluation for Shogi (Phase 2: Material + Positional).
//!
//! This evaluation combines two components:
//!
//!   1. **Material**: The total value of pieces on the board and in hand.
//!      A Rook is worth more than a Pawn, etc.
//!
//!   2. **Piece-Square Tables (PST)**: A bonus or penalty for each piece
//!      depending on which square it occupies. This captures basic positional
//!      knowledge — for example:
//!        - Pawns are better when advanced (closer to promotion)
//!        - The King should be in a safe castle position (corner)
//!        - Rooks are strong on open files in the center
//!        - Gold and Silver are strong when defending the King
//!
//! The PST values are defined from Black's perspective (Black starts at the
//! bottom, rank 8, and advances toward rank 0). For White's pieces, the table
//! is flipped vertically (rank 0 ↔ rank 8, rank 1 ↔ rank 7, etc.) so that
//! both sides use symmetric positional logic.
//!
//! === How to read the PST tables ===
//!
//! Each table is a 9×9 array indexed as [rank][file], where:
//!   - rank 0 = top of the board (White's back rank)
//!   - rank 8 = bottom of the board (Black's back rank)
//!   - file 0 = right side (file "1" in Shogi notation)
//!   - file 8 = left side (file "9" in Shogi notation)
//!
//! Values are in centipawns. Positive = good position for the piece.
//! These are intentionally rough — the NNUE will learn precise values.

use shogi::{Color, Move, Piece, PieceType, Position, Square};

// ============================================================================
// Evaluator trait
// ============================================================================

/// Trait for position evaluation functions.
///
/// This trait is designed so that evaluators that maintain incremental state
/// (like NNUE's accumulator) can update cheaply as the search makes and
/// unmakes moves. Stateless evaluators (like PST) simply ignore the hooks.
///
/// Contract expected by the search driver:
///
/// 1. `evaluate(pos)` must return the correct score for the current `pos`.
/// 2. Before every `pos.make_move(m)`, the search calls
///    `on_make(pos_before, m)`. The evaluator may snapshot any internal
///    state and apply a forward update.
/// 3. If `make_move` fails (e.g. illegal move), the search calls
///    `on_unmake()` immediately to roll back the pre-emptive update.
/// 4. If `make_move` succeeds and the search later `unmake_move`s, it
///    calls `on_unmake()` to restore the evaluator to the pre-move state.
/// 5. `refresh(pos)` discards all incremental state and resyncs to `pos`.
///    The search uses this at the root of each new position (e.g. the start
///    of a game, or when re-entering the searcher after a move was played
///    on the game board).
pub trait Evaluator: Send {
    /// Evaluate the current position from the side-to-move's perspective.
    /// Returns centipawns (positive = good for side-to-move).
    fn evaluate(&mut self, pos: &Position) -> i32;

    /// Hook called before `pos.make_move(m)` is executed.
    /// Default: no-op (for stateless evaluators).
    fn on_make(&mut self, _pos_before: &Position, _m: &Move) {}

    /// Hook called after `pos.unmake_move()` (or after a failed make_move,
    /// to roll back a pre-emptive `on_make`). Default: no-op.
    fn on_unmake(&mut self) {}

    /// Discard all incremental state and resync to `pos` from scratch.
    /// Default: no-op (for stateless evaluators).
    fn refresh(&mut self, _pos: &Position) {}
}

/// The built-in material + piece-square-table evaluator (Phase 2).
///
/// PST evaluation is stateless — every call recomputes from the position,
/// so the make/unmake hooks are no-ops.
#[derive(Default, Clone, Copy)]
pub struct PstEvaluator;

impl Evaluator for PstEvaluator {
    fn evaluate(&mut self, pos: &Position) -> i32 {
        evaluate(pos)
    }
}

// ============================================================================
// Material values (centipawns)
// ============================================================================

/// Base material value of each piece type.
fn piece_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn => 100,
        PieceType::Lance => 350,
        PieceType::Knight => 400,
        PieceType::Silver => 450,
        PieceType::Gold => 550,
        PieceType::Bishop => 700,
        PieceType::Rook => 900,
        PieceType::ProPawn => 550,
        PieceType::ProLance => 550,
        PieceType::ProKnight => 550,
        PieceType::ProSilver => 550,
        PieceType::ProBishop => 900,
        PieceType::ProRook => 1100,
        PieceType::King => 0,
    }
}

// ============================================================================
// Piece-Square Tables
// ============================================================================
//
// All tables are from Black's perspective (advancing toward rank 0).
// White's values are obtained by flipping the rank: rank → 8 - rank.
//
// Layout: PST[rank][file], 9 ranks × 9 files.
// Values are small relative to material (typically ±5 to ±30 cp) so that
// positional factors influence the evaluation without overriding captures.

/// Pawn PST: Pawns are better when advanced. Central pawns are slightly
/// more valuable. Back-rank pawns can't exist (must promote), so rank 0
/// values don't matter much.
#[rustfmt::skip]
const PAWN_PST: [[i32; 9]; 9] = [
    //  file: 9    8    7    6    5    4    3    2    1
    [   0,   0,   0,   0,   0,   0,   0,   0,   0], // rank 0 (opponent's back rank)
    [   0,   0,   0,   0,   0,   0,   0,   0,   0], // rank 1
    [   0,   0,   0,   0,   0,   0,   0,   0,   0], // rank 2 (promotion zone starts)
    [   5,   5,  10,  10,  15,  10,  10,   5,   5], // rank 3
    [   3,   3,   5,   8,  10,   8,   5,   3,   3], // rank 4
    [   2,   2,   3,   5,   8,   5,   3,   2,   2], // rank 5
    [   0,   0,   0,   3,   5,   3,   0,   0,   0], // rank 6 (starting rank for Black pawns)
    [   0,   0,   0,   0,   0,   0,   0,   0,   0], // rank 7
    [   0,   0,   0,   0,   0,   0,   0,   0,   0], // rank 8 (Black's back rank)
];

/// Lance PST: Lances are most effective on open files. Slight preference
/// for staying on the back ranks where they support an attack column.
#[rustfmt::skip]
const LANCE_PST: [[i32; 9]; 9] = [
    [   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [   5,   5,   5,   5,   5,   5,   5,   5,   5],
    [   3,   3,   3,   3,   3,   3,   3,   3,   3],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [   0,   0,   3,   3,   3,   3,   3,   0,   0],
    [   0,   0,   5,   5,   5,   5,   5,   0,   0],
];

/// Knight PST: Knights have very limited movement in Shogi (only 2 forward
/// jumps). They're most useful when advanced into the opponent's territory,
/// but vulnerable if unsupported. Penalize edge placement.
#[rustfmt::skip]
const KNIGHT_PST: [[i32; 9]; 9] = [
    [   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  -5,   5,  10,  15,  15,  15,  10,   5,  -5],
    [  -5,   5,  10,  10,  10,  10,  10,   5,  -5],
    [  -5,   0,   5,   5,   5,   5,   5,   0,  -5],
    [  -5,   0,   0,   3,   3,   3,   0,   0,  -5],
    [  -5,   0,   0,   0,   0,   0,   0,   0,  -5],
    [ -10,  -5,   0,   0,   0,   0,   0,  -5, -10],
    [ -10,  -5,   0,   0,   0,   0,   0,  -5, -10],
];

/// Silver PST: Silvers are versatile attackers. They move diagonally forward
/// and straight forward, making them excellent for attacking. Prefer central
/// and advanced positions.
#[rustfmt::skip]
const SILVER_PST: [[i32; 9]; 9] = [
    [   0,   0,   5,  10,  10,  10,   5,   0,   0],
    [   0,   5,  10,  15,  15,  15,  10,   5,   0],
    [   0,   5,  10,  15,  15,  15,  10,   5,   0],
    [   0,   3,   5,  10,  10,  10,   5,   3,   0],
    [   0,   0,   3,   5,   5,   5,   3,   0,   0],
    [   0,   0,   0,   3,   5,   3,   0,   0,   0],
    [   0,   0,   0,   0,   3,   0,   0,   0,   0],
    [   5,   5,   0,   0,   0,   0,   0,   5,   5],
    [   5,   5,   0,   0,   0,   0,   0,   5,   5],
];

/// Gold PST: Golds are the primary defensive piece. They're most valuable
/// near the King (typically in a castle formation). Also useful for attacking
/// when advanced, but generally best kept near the King.
#[rustfmt::skip]
const GOLD_PST: [[i32; 9]; 9] = [
    [   0,   5,  10,  10,  10,  10,  10,   5,   0],
    [   0,   5,  10,  15,  15,  15,  10,   5,   0],
    [   0,   5,  10,  10,  10,  10,  10,   5,   0],
    [   0,   0,   5,   5,   5,   5,   5,   0,   0],
    [   0,   0,   0,   3,   3,   3,   0,   0,   0],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [   5,   3,   0,   0,   0,   0,   0,   3,   5],
    [   5,   5,   5,   3,   3,   3,   5,   5,   5],
    [   5,  10,   5,   5,   3,   5,   5,  10,   5],
];

/// Bishop PST: Bishops are long-range diagonal pieces. They're strong from
/// central positions where diagonals are longest. The standard "bishop on
/// 8h/2b" opening position is good, but the center is even better.
#[rustfmt::skip]
const BISHOP_PST: [[i32; 9]; 9] = [
    [  -5,   0,   0,   5,  10,   5,   0,   0,  -5],
    [   0,   0,   5,  10,  10,  10,   5,   0,   0],
    [   0,   5,  10,  10,  15,  10,  10,   5,   0],
    [   5,  10,  10,  15,  15,  15,  10,  10,   5],
    [   0,   5,  10,  15,  20,  15,  10,   5,   0],
    [   0,   5,  10,  10,  15,  10,  10,   5,   0],
    [   0,   0,   5,  10,  10,  10,   5,   0,   0],
    [   0,   0,   0,   5,   5,   5,   0,   0,   0],
    [  -5,   0,   0,   0,   5,   0,   0,   0,  -5],
];

/// Rook PST: Rooks are long-range orthogonal pieces. They're strongest on
/// open files (no own Pawns blocking). Central files are preferred. Slight
/// bonus for the starting position (file 2/8 — the "Rook file").
#[rustfmt::skip]
const ROOK_PST: [[i32; 9]; 9] = [
    [   5,   5,   5,  10,  10,  10,   5,   5,   5],
    [   5,   5,   5,  10,  15,  10,   5,   5,   5],
    [   5,   5,   5,  10,  15,  10,   5,   5,   5],
    [   3,   3,   5,  10,  10,  10,   5,   3,   3],
    [   0,   0,   3,   5,  10,   5,   3,   0,   0],
    [   0,   0,   0,   3,   5,   3,   0,   0,   0],
    [   0,   0,   0,   0,   3,   0,   0,   0,   0],
    [   0,   5,   0,   0,   0,   0,   0,   5,   0],
    [   0,   5,   0,   0,   0,   0,   0,   5,   0],
];

/// King PST: The King should be in a safe castle position. In Shogi, the
/// most common castles put the King in the corner (files 1-3 or 7-9) on
/// ranks 7-8. The center is dangerous. Advanced Kings are risky but can
/// be strong in endgames.
///
/// This table strongly encourages castling to the sides and penalizes
/// the center and exposed positions.
#[rustfmt::skip]
const KING_PST: [[i32; 9]; 9] = [
    [ -30, -20, -20, -30, -40, -30, -20, -20, -30],
    [ -20, -10, -10, -20, -30, -20, -10, -10, -20],
    [ -10,  -5,  -5, -10, -20, -10,  -5,  -5, -10],
    [ -10,  -5,  -5, -10, -15, -10,  -5,  -5, -10],
    [ -10,  -5,  -5, -10, -10, -10,  -5,  -5, -10],
    [  -5,   0,   0,  -5,  -5,  -5,   0,   0,  -5],
    [   0,   5,   5,   0,  -5,   0,   5,   5,   0],
    [  10,  15,  10,   0, -10,   0,  10,  15,  10],
    [  15,  20,  15,   5, -15,   5,  15,  20,  15],
];

/// Promoted pieces PST (shared by ProPawn, ProLance, ProKnight, ProSilver).
/// Promoted pieces that move like Gold are best when advanced into the
/// opponent's territory where they can attack.
#[rustfmt::skip]
const PROMOTED_GOLD_PST: [[i32; 9]; 9] = [
    [   5,  10,  15,  15,  15,  15,  15,  10,   5],
    [   5,  10,  15,  20,  20,  20,  15,  10,   5],
    [   5,  10,  15,  15,  15,  15,  15,  10,   5],
    [   0,   5,  10,  10,  10,  10,  10,   5,   0],
    [   0,   0,   5,   5,   5,   5,   5,   0,   0],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0],
];

/// Horse (promoted Bishop) PST: The Horse is extremely powerful — it has
/// Bishop movement plus King movement. Best in the center where it controls
/// maximum squares.
#[rustfmt::skip]
const HORSE_PST: [[i32; 9]; 9] = [
    [   0,   5,  10,  15,  20,  15,  10,   5,   0],
    [   5,  10,  15,  20,  25,  20,  15,  10,   5],
    [  10,  15,  20,  25,  30,  25,  20,  15,  10],
    [   5,  10,  15,  20,  25,  20,  15,  10,   5],
    [   0,   5,  10,  15,  20,  15,  10,   5,   0],
    [   0,   0,   5,  10,  15,  10,   5,   0,   0],
    [   0,   0,   0,   5,  10,   5,   0,   0,   0],
    [   0,   0,   0,   0,   5,   0,   0,   0,   0],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0],
];

/// Dragon (promoted Rook) PST: The Dragon is the strongest piece — Rook
/// movement plus King movement. Like the Horse, best in the center and
/// advanced positions.
#[rustfmt::skip]
const DRAGON_PST: [[i32; 9]; 9] = [
    [   5,  10,  10,  15,  20,  15,  10,  10,   5],
    [  10,  10,  15,  20,  25,  20,  15,  10,  10],
    [  10,  15,  20,  25,  30,  25,  20,  15,  10],
    [   5,  10,  15,  20,  25,  20,  15,  10,   5],
    [   0,   5,  10,  15,  20,  15,  10,   5,   0],
    [   0,   0,   5,  10,  15,  10,   5,   0,   0],
    [   0,   0,   0,   5,  10,   5,   0,   0,   0],
    [   0,   0,   0,   0,   5,   0,   0,   0,   0],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0],
];

/// Look up the PST table for a given piece type.
fn pst_table(pt: PieceType) -> &'static [[i32; 9]; 9] {
    match pt {
        PieceType::Pawn => &PAWN_PST,
        PieceType::Lance => &LANCE_PST,
        PieceType::Knight => &KNIGHT_PST,
        PieceType::Silver => &SILVER_PST,
        PieceType::Gold => &GOLD_PST,
        PieceType::Bishop => &BISHOP_PST,
        PieceType::Rook => &ROOK_PST,
        PieceType::King => &KING_PST,
        PieceType::ProPawn | PieceType::ProLance | PieceType::ProKnight | PieceType::ProSilver => {
            &PROMOTED_GOLD_PST
        }
        PieceType::ProBishop => &HORSE_PST,
        PieceType::ProRook => &DRAGON_PST,
    }
}

/// Look up the positional bonus for a piece on a square.
///
/// Tables are defined from Black's perspective. For White, we flip the rank
/// (rank → 8 - rank) so that White's "forward" direction is handled correctly.
fn pst_bonus(pt: PieceType, sq: Square, color: Color) -> i32 {
    let table = pst_table(pt);
    let file = sq.file() as usize;
    let rank = match color {
        Color::Black => sq.rank() as usize,
        Color::White => 8 - sq.rank() as usize, // Flip for White's perspective
    };
    table[rank][file]
}

// ============================================================================
// Main evaluation function
// ============================================================================

/// Evaluate a position from the side-to-move's perspective.
///
/// Returns a score in centipawns:
///   - Positive = side-to-move has a better position
///   - Negative = opponent has a better position
///   - Zero = approximately equal
///
/// The evaluation considers:
///   1. **Material**: Total piece values on the board and in hand
///   2. **Piece-square bonuses**: Positional value based on where pieces sit
pub fn evaluate(pos: &Position) -> i32 {
    let stm = pos.side_to_move();
    let mut score: i32 = 0;

    // --- Board pieces: material + positional ---
    for i in 0..81 {
        let sq = Square::from_index(i).unwrap();
        if let Some(piece) = pos.piece_at(sq) {
            let material = piece_value(piece.piece_type);
            let positional = pst_bonus(piece.piece_type, sq, piece.color);
            let total = material + positional;

            if piece.color == stm {
                score += total;
            } else {
                score -= total;
            }
        }
    }

    // --- Hand pieces: material only (no positional bonus) ---
    // Hand pieces aren't on the board, so they don't get PST bonuses.
    // They still have material value because they can be dropped.
    let hand_types = [
        PieceType::Pawn,
        PieceType::Lance,
        PieceType::Knight,
        PieceType::Silver,
        PieceType::Gold,
        PieceType::Bishop,
        PieceType::Rook,
    ];

    for &pt in &hand_types {
        let value = piece_value(pt);

        let our_piece = Piece {
            piece_type: pt,
            color: stm,
        };
        let our_count = pos.hand(our_piece) as i32;
        score += value * our_count;

        let their_piece = Piece {
            piece_type: pt,
            color: stm.flip(),
        };
        let their_count = pos.hand(their_piece) as i32;
        score -= value * their_count;
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;
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
    // piece_value — sanity checks
    // -----------------------------------------------------------------------

    #[test]
    fn material_values_are_positive_except_king() {
        let types = [
            PieceType::Pawn,
            PieceType::Lance,
            PieceType::Knight,
            PieceType::Silver,
            PieceType::Gold,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::ProPawn,
            PieceType::ProLance,
            PieceType::ProKnight,
            PieceType::ProSilver,
            PieceType::ProBishop,
            PieceType::ProRook,
        ];
        for pt in types {
            assert!(piece_value(pt) > 0, "{:?} should have positive value", pt);
        }
        assert_eq!(piece_value(PieceType::King), 0);
    }

    #[test]
    fn promoted_piece_at_least_as_valuable_as_base() {
        assert!(piece_value(PieceType::ProPawn) >= piece_value(PieceType::Pawn));
        assert!(piece_value(PieceType::ProLance) >= piece_value(PieceType::Lance));
        assert!(piece_value(PieceType::ProKnight) >= piece_value(PieceType::Knight));
        assert!(piece_value(PieceType::ProSilver) >= piece_value(PieceType::Silver));
        assert!(piece_value(PieceType::ProBishop) >= piece_value(PieceType::Bishop));
        assert!(piece_value(PieceType::ProRook) >= piece_value(PieceType::Rook));
    }

    #[test]
    fn rook_more_valuable_than_bishop() {
        assert!(piece_value(PieceType::Rook) > piece_value(PieceType::Bishop));
    }

    // -----------------------------------------------------------------------
    // pst_bonus — basic checks
    // -----------------------------------------------------------------------

    #[test]
    fn pst_bonus_black_white_symmetry() {
        // For the same piece on symmetric squares, Black's bonus at rank R
        // should equal White's bonus at rank 8-R.
        let sq_black = Square::new(4, 6).unwrap(); // file 4, rank 6
        let sq_white = Square::new(4, 2).unwrap(); // file 4, rank 2 (= 8-6)
        let bonus_b = pst_bonus(PieceType::Pawn, sq_black, Color::Black);
        let bonus_w = pst_bonus(PieceType::Pawn, sq_white, Color::White);
        assert_eq!(bonus_b, bonus_w);
    }

    // -----------------------------------------------------------------------
    // evaluate — integration
    // -----------------------------------------------------------------------

    #[test]
    fn startpos_is_roughly_balanced() {
        let pos = make_pos("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
        let score = evaluate(&pos);
        // Starting position should be nearly equal (PST bonuses may cause small offset).
        assert!(
            score.abs() < 50,
            "startpos should be roughly balanced, got {score}"
        );
    }

    #[test]
    fn extra_rook_is_large_advantage() {
        // Black has an extra Rook in hand.
        let pos = make_pos("4k4/9/9/9/9/9/9/9/4K4 b R 1");
        let score = evaluate(&pos);
        assert!(
            score >= 900,
            "extra Rook should give ~900cp advantage, got {score}"
        );
    }

    #[test]
    fn material_disadvantage_gives_negative_score() {
        // White has an extra Rook in hand.
        let pos = make_pos("4k4/9/9/9/9/9/9/9/4K4 b r 1");
        let score = evaluate(&pos);
        assert!(
            score <= -900,
            "opponent's extra Rook should be negative, got {score}"
        );
    }

    #[test]
    fn evaluate_returns_stm_relative_score() {
        // Same position, flip side to move. Score sign should flip.
        let pos_b = make_pos("4k4/9/9/9/9/9/9/9/4K4 b R 1");
        let pos_w = make_pos("4k4/9/9/9/9/9/9/9/4K4 w R 1");
        let score_b = evaluate(&pos_b);
        let score_w = evaluate(&pos_w);
        // Black has extra Rook: positive for Black, negative for White.
        assert!(score_b > 0);
        assert!(score_w < 0);
        assert_eq!(score_b, -score_w);
    }

    #[test]
    fn pst_evaluator_trait_matches_fn() {
        let pos = make_pos("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
        let eval_fn = evaluate(&pos);
        let mut pst = PstEvaluator;
        let eval_trait = pst.evaluate(&pos);
        assert_eq!(eval_fn, eval_trait);
    }
}
