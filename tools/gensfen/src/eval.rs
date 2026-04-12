//! Material-only position evaluation for Shogi.
//!
//! Phase 1 evaluation: simply count material (piece values) for both sides.
//! This is the simplest useful evaluation — it knows that a Rook is worth more
//! than a Pawn, but it knows nothing about piece positions, king safety, etc.
//!
//! Despite its simplicity, material evaluation + alpha-beta search produces
//! reasonable training data because the search discovers tactical sequences
//! (captures, exchanges) that change the material balance.

use shogi::{Piece, PieceType, Position, Square};

/// Piece values in centipawns (cp).
///
/// These are standard Shogi piece values used in most engines.
/// 100cp = 1 Pawn of advantage.
///
/// Promoted pieces that move like Gold (ProPawn, ProLance, ProKnight, ProSilver)
/// are valued slightly above Gold because promotion is an advantage — the
/// opponent can only capture the unpromoted form.
fn piece_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn => 100,
        PieceType::Lance => 350,
        PieceType::Knight => 400,
        PieceType::Silver => 450,
        PieceType::Gold => 550,
        PieceType::Bishop => 700,
        PieceType::Rook => 900,
        // Promoted pieces that move like Gold
        PieceType::ProPawn => 550,
        PieceType::ProLance => 550,
        PieceType::ProKnight => 550,
        PieceType::ProSilver => 550,
        // Horse = Bishop + King movement, very strong
        PieceType::ProBishop => 900,
        // Dragon = Rook + King movement, strongest non-King piece
        PieceType::ProRook => 1100,
        // King is not counted in material (both sides always have exactly one)
        PieceType::King => 0,
    }
}

/// Evaluate a position from the side-to-move's perspective.
///
/// Returns a score in centipawns:
///   - Positive = side-to-move has more material (good position)
///   - Negative = opponent has more material (bad position)
///   - Zero = material is equal
///
/// The evaluation considers:
///   1. Pieces on the board (all 81 squares)
///   2. Pieces in hand (captured pieces available for dropping)
pub fn evaluate(pos: &Position) -> i32 {
    let stm = pos.side_to_move();
    let mut score: i32 = 0;

    // --- Board pieces ---
    // Iterate over all 81 squares and sum up piece values.
    for i in 0..81 {
        let sq = Square::from_index(i).unwrap();
        if let Some(piece) = pos.piece_at(sq) {
            let value = piece_value(piece.piece_type);
            if piece.color == stm {
                score += value; // Our piece: positive contribution
            } else {
                score -= value; // Opponent's piece: negative contribution
            }
        }
    }

    // --- Hand pieces ---
    // In Shogi, captured pieces go to your "hand" and can be dropped back
    // onto the board. Hand pieces have the same material value as board pieces.
    // Only unpromoted non-King pieces can be in hand.
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

        // Count pieces in our hand
        let our_piece = Piece { piece_type: pt, color: stm };
        let our_count = pos.hand(our_piece) as i32;
        score += value * our_count;

        // Count pieces in opponent's hand
        let their_piece = Piece { piece_type: pt, color: stm.flip() };
        let their_count = pos.hand(their_piece) as i32;
        score -= value * their_count;
    }

    score
}
