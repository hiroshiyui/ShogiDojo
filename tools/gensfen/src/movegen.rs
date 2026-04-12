//! Pseudo-legal move generation for Shogi.
//!
//! This module generates candidate moves from a position. The moves are
//! "pseudo-legal" — they obey piece movement rules and promotion rules,
//! but some may leave the king in check (which makes them illegal).
//!
//! We rely on `Position::make_move()` in the search to filter out truly
//! illegal moves. This is the standard approach in chess/shogi engines:
//! generate fast, validate during search.
//!
//! Move types in Shogi:
//!   1. **Normal moves**: A piece on the board moves to another square.
//!      May capture an opponent's piece. May promote if entering/leaving
//!      the promotion zone (last 3 ranks from the player's perspective).
//!   2. **Drop moves**: A captured piece from your hand is placed on any
//!      empty square (with restrictions: no Pawns/Lances on the last rank,
//!      no Knights on the last two ranks, no two unpromoted Pawns on the
//!      same file — "nifu" rule).

use shogi::{Color, Move, Piece, PieceType, Position, Square};

/// Generate all pseudo-legal moves for the side to move.
///
/// Returns a Vec of moves. Some may be illegal (leaving king in check,
/// nifu for Pawn drops, etc.) — the search handles these via `make_move()`
/// error returns.
pub fn generate_moves(pos: &Position) -> Vec<Move> {
    let stm = pos.side_to_move();
    let mut moves = Vec::with_capacity(128);

    generate_board_moves(pos, stm, &mut moves);
    generate_drop_moves(pos, stm, &mut moves);

    moves
}

/// Generate moves for pieces already on the board (normal moves).
fn generate_board_moves(pos: &Position, stm: Color, moves: &mut Vec<Move>) {
    for i in 0..81 {
        let from = Square::from_index(i).unwrap();
        let piece = match pos.piece_at(from) {
            Some(p) if p.color == stm => *p,
            _ => continue, // Empty square or opponent's piece
        };

        // Get the bitboard of squares this piece can move to.
        // This handles the piece's movement rules but doesn't check for
        // leaving our own king in check.
        let targets = pos.move_candidates(from, piece);

        for to in targets {
            let can_promote = can_piece_promote(piece.piece_type);
            let in_promo_zone =
                in_promotion_zone(from, stm) || in_promotion_zone(to, stm);

            if can_promote && in_promo_zone {
                // When promotion is possible, generate BOTH the promoted and
                // non-promoted versions. The search can try both.
                moves.push(Move::Normal { from, to, promote: true });

                // However, some pieces MUST promote if they can't retreat:
                //   - Pawn/Lance on the last rank
                //   - Knight on the last two ranks
                if !must_promote(piece.piece_type, to, stm) {
                    moves.push(Move::Normal { from, to, promote: false });
                }
            } else {
                moves.push(Move::Normal { from, to, promote: false });
            }
        }
    }
}

/// Generate drop moves (placing captured pieces from hand onto empty squares).
fn generate_drop_moves(pos: &Position, stm: Color, moves: &mut Vec<Move>) {
    // Piece types that can be dropped (all hand pieces are unpromoted)
    let drop_types = [
        PieceType::Pawn,
        PieceType::Lance,
        PieceType::Knight,
        PieceType::Silver,
        PieceType::Gold,
        PieceType::Bishop,
        PieceType::Rook,
    ];

    for &pt in &drop_types {
        let piece = Piece { piece_type: pt, color: stm };
        if pos.hand(piece) == 0 {
            continue; // Don't have this piece in hand
        }

        // Drop onto every empty square (with placement restrictions)
        for i in 0..81 {
            let to = Square::from_index(i).unwrap();

            // Square must be empty
            if pos.piece_at(to).is_some() {
                continue;
            }

            // Placement restrictions:
            // Pawn and Lance cannot be dropped on the last rank (nowhere to go)
            // Knight cannot be dropped on the last two ranks (nowhere to go)
            if !piece.is_placeable_at(to) {
                continue;
            }

            // Note: the "nifu" rule (no two unpromoted Pawns on the same file)
            // is NOT checked here — make_move() catches it and returns an error.
            moves.push(Move::Drop { to, piece_type: pt });
        }
    }
}

/// Can this piece type promote? Kings and Golds cannot. Already-promoted
/// pieces cannot promote again.
fn can_piece_promote(pt: PieceType) -> bool {
    matches!(
        pt,
        PieceType::Pawn
            | PieceType::Lance
            | PieceType::Knight
            | PieceType::Silver
            | PieceType::Bishop
            | PieceType::Rook
    )
}

/// Must this piece type promote when landing on `to`?
///
/// Pieces that cannot retreat from their destination MUST promote:
///   - Pawn and Lance on the last rank (rank 0 for Black, rank 8 for White)
///   - Knight on the last two ranks (ranks 0-1 for Black, ranks 7-8 for White)
fn must_promote(pt: PieceType, to: Square, color: Color) -> bool {
    let rank = to.rank();
    match (pt, color) {
        // Black moves toward rank 0 (top of board)
        (PieceType::Pawn | PieceType::Lance, Color::Black) => rank == 0,
        (PieceType::Knight, Color::Black) => rank <= 1,
        // White moves toward rank 8 (bottom of board)
        (PieceType::Pawn | PieceType::Lance, Color::White) => rank == 8,
        (PieceType::Knight, Color::White) => rank >= 7,
        _ => false,
    }
}

/// Is this square in the promotion zone for the given color?
///
/// The promotion zone is the furthest 3 ranks from the player's starting side:
///   - Black's promotion zone: ranks 0, 1, 2 (top 3 rows — opponent's territory)
///   - White's promotion zone: ranks 6, 7, 8 (bottom 3 rows — opponent's territory)
fn in_promotion_zone(sq: Square, color: Color) -> bool {
    let rank = sq.rank();
    match color {
        Color::Black => rank <= 2,
        Color::White => rank >= 6,
    }
}
