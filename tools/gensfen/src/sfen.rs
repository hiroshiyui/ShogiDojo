//! Build a standalone SFEN string from the current board state.
//!
//! shogi-rs's `Position::to_sfen()` returns the initial position + move list
//! when there's a move history (e.g., "lnsgkgsnl/... b - 1 moves 7g7f 3c3d").
//! Our Python training pipeline expects a **standalone** SFEN that describes
//! the current board state directly (no move list), so we build it ourselves.
//!
//! This replicates the private `Position::generate_sfen()` method from shogi-rs.

use shogi::{Color, Piece, PieceType, Position, Square};

/// Build a 4-field SFEN string for the current board state.
///
/// Format: `<board> <side> <hand> <ply>`
///
/// Example: `lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1`
pub fn position_to_sfen(pos: &Position) -> String {
    // --- Board ---
    // Iterate rank by rank (row 0 = top, row 8 = bottom).
    // Within each rank, iterate files from 8 (file "9") down to 0 (file "1").
    // This matches the standard SFEN board notation.
    let board: String = (0..9u8)
        .map(|rank| {
            let mut row = String::new();
            let mut empty_count = 0;

            for file in (0..9u8).rev() {
                let sq = Square::new(file, rank).unwrap();
                match pos.piece_at(sq) {
                    Some(piece) => {
                        if empty_count > 0 {
                            row.push_str(&empty_count.to_string());
                            empty_count = 0;
                        }
                        row.push_str(&piece.to_string());
                    }
                    None => {
                        empty_count += 1;
                    }
                }
            }

            if empty_count > 0 {
                row.push_str(&empty_count.to_string());
            }
            row
        })
        .collect::<Vec<_>>()
        .join("/");

    // --- Side to move ---
    let side = if pos.side_to_move() == Color::Black {
        "b"
    } else {
        "w"
    };

    // --- Hand pieces ---
    // Format: Black's hand pieces first (uppercase), then White's (lowercase).
    // Pieces are listed in order: Rook, Bishop, Gold, Silver, Knight, Lance, Pawn.
    // Count > 1 is prefixed as a number (e.g., "2P" = 2 Pawns).
    // If no pieces in hand: "-".
    let hand_piece_order = [
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Gold,
        PieceType::Silver,
        PieceType::Knight,
        PieceType::Lance,
        PieceType::Pawn,
    ];

    let mut hand = String::new();
    for &color in &[Color::Black, Color::White] {
        for &pt in &hand_piece_order {
            let piece = Piece {
                piece_type: pt,
                color,
            };
            let count = pos.hand(piece);
            if count == 0 {
                continue;
            }
            if count > 1 {
                hand.push_str(&count.to_string());
            }
            hand.push_str(&piece.to_string());
        }
    }

    if hand.is_empty() {
        hand = "-".to_string();
    }

    // --- Ply (move number) ---
    // Use the ply counter from the position.
    let ply = pos.ply();

    format!("{board} {side} {hand} {ply}")
}
