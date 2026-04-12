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

use shogi::{Color, Piece, PieceType, Position, Square};

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

        let our_piece = Piece { piece_type: pt, color: stm };
        let our_count = pos.hand(our_piece) as i32;
        score += value * our_count;

        let their_piece = Piece { piece_type: pt, color: stm.flip() };
        let their_count = pos.hand(their_piece) as i32;
        score -= value * their_count;
    }

    score
}
