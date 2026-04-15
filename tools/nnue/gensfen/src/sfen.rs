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

#[cfg(test)]
mod tests {
    use super::*;
    use shogi::bitboard::Factory as BBFactory;

    fn init() {
        // BBFactory::init() is idempotent but must be called before any board ops.
        BBFactory::init();
    }

    fn make_pos(sfen: &str) -> Position {
        init();
        let mut pos = Position::new();
        pos.set_sfen(sfen).unwrap();
        pos
    }

    #[test]
    fn startpos_sfen() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let pos = make_pos(sfen);
        assert_eq!(position_to_sfen(&pos), sfen);
    }

    #[test]
    fn side_to_move_white() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 2";
        let pos = make_pos(sfen);
        let result = position_to_sfen(&pos);
        let parts: Vec<&str> = result.split(' ').collect();
        assert_eq!(parts[1], "w");
    }

    #[test]
    fn empty_board_with_kings() {
        let sfen = "4k4/9/9/9/9/9/9/9/4K4 b - 1";
        let pos = make_pos(sfen);
        assert_eq!(position_to_sfen(&pos), sfen);
    }

    #[test]
    fn hand_pieces_formatted() {
        // Black has 2 Pawns and a Bishop; White has a Gold.
        let sfen = "4k4/9/9/9/9/9/9/9/4K4 b B2Pg 1";
        let pos = make_pos(sfen);
        let result = position_to_sfen(&pos);
        let parts: Vec<&str> = result.split(' ').collect();
        // Hand should contain B, 2P (Black) and g (White).
        let hand = parts[2];
        assert!(
            hand.contains('B'),
            "hand should include Black's Bishop: {hand}"
        );
        assert!(
            hand.contains('P'),
            "hand should include Black's Pawn: {hand}"
        );
        assert!(
            hand.contains('g'),
            "hand should include White's Gold: {hand}"
        );
    }

    #[test]
    fn no_hand_is_dash() {
        let sfen = "4k4/9/9/9/9/9/9/9/4K4 b - 1";
        let pos = make_pos(sfen);
        let result = position_to_sfen(&pos);
        let parts: Vec<&str> = result.split(' ').collect();
        assert_eq!(parts[2], "-");
    }

    #[test]
    fn promoted_piece_in_board() {
        // Black promoted Bishop (Horse) on center square.
        let sfen = "4k4/9/9/9/4+B4/9/9/9/4K4 b - 1";
        let pos = make_pos(sfen);
        let result = position_to_sfen(&pos);
        assert!(
            result.contains("+B"),
            "SFEN should contain promoted Bishop: {result}"
        );
    }

    #[test]
    fn sfen_has_four_fields() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let pos = make_pos(sfen);
        let result = position_to_sfen(&pos);
        assert_eq!(result.split(' ').count(), 4);
    }
}
