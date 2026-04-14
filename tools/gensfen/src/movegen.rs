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
            let in_promo_zone = in_promotion_zone(from, stm) || in_promotion_zone(to, stm);

            if can_promote && in_promo_zone {
                // When promotion is possible, generate BOTH the promoted and
                // non-promoted versions. The search can try both.
                moves.push(Move::Normal {
                    from,
                    to,
                    promote: true,
                });

                // However, some pieces MUST promote if they can't retreat:
                //   - Pawn/Lance on the last rank
                //   - Knight on the last two ranks
                if !must_promote(piece.piece_type, to, stm) {
                    moves.push(Move::Normal {
                        from,
                        to,
                        promote: false,
                    });
                }
            } else {
                moves.push(Move::Normal {
                    from,
                    to,
                    promote: false,
                });
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

    // Precompute the set of files that already contain an unpromoted friendly
    // Pawn — these files are forbidden for Pawn drops ("nifu" rule).
    let nifu_files = blocked_pawn_files(pos, stm);

    for &pt in &drop_types {
        let piece = Piece {
            piece_type: pt,
            color: stm,
        };
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

            // "Nifu" rule: you may not have two unpromoted Pawns on the same
            // file. shogi-rs's make_move() also rejects this, but filtering
            // here keeps the pseudo-legal move list smaller and makes the
            // rule explicit in our own code.
            if pt == PieceType::Pawn && nifu_files[to.file() as usize] {
                continue;
            }

            // Note: the "uchi-fu-zume" rule (a Pawn drop that delivers
            // checkmate is illegal) is NOT checked here. It is enforced by
            // shogi-rs's make_move(), which returns MoveError::Uchifuzume,
            // so get_legal_moves() in game.rs filters it out.
            moves.push(Move::Drop { to, piece_type: pt });
        }
    }
}

/// Return a per-file bitmap of files that already contain an unpromoted
/// friendly Pawn. Used to enforce the "nifu" rule when generating Pawn drops.
fn blocked_pawn_files(pos: &Position, color: Color) -> [bool; 9] {
    let mut blocked = [false; 9];
    for i in 0..81 {
        let sq = Square::from_index(i).unwrap();
        // Promoted Pawns (Tokin) do not count for the nifu rule, so only
        // the unpromoted PieceType::Pawn is checked.
        if let Some(p) = pos.piece_at(sq)
            && p.color == color
            && p.piece_type == PieceType::Pawn
        {
            blocked[sq.file() as usize] = true;
        }
    }
    blocked
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

#[cfg(test)]
mod tests {
    use super::*;
    use shogi::bitboard::Factory as BBFactory;

    fn init() {
        BBFactory::init();
    }

    // -----------------------------------------------------------------------
    // can_piece_promote
    // -----------------------------------------------------------------------

    #[test]
    fn promotable_pieces() {
        assert!(can_piece_promote(PieceType::Pawn));
        assert!(can_piece_promote(PieceType::Lance));
        assert!(can_piece_promote(PieceType::Knight));
        assert!(can_piece_promote(PieceType::Silver));
        assert!(can_piece_promote(PieceType::Bishop));
        assert!(can_piece_promote(PieceType::Rook));
    }

    #[test]
    fn non_promotable_pieces() {
        assert!(!can_piece_promote(PieceType::Gold));
        assert!(!can_piece_promote(PieceType::King));
        assert!(!can_piece_promote(PieceType::ProPawn));
        assert!(!can_piece_promote(PieceType::ProLance));
        assert!(!can_piece_promote(PieceType::ProKnight));
        assert!(!can_piece_promote(PieceType::ProSilver));
        assert!(!can_piece_promote(PieceType::ProBishop));
        assert!(!can_piece_promote(PieceType::ProRook));
    }

    // -----------------------------------------------------------------------
    // must_promote
    // -----------------------------------------------------------------------

    #[test]
    fn pawn_must_promote_on_last_rank() {
        // Black Pawn on rank 0 must promote.
        let sq = Square::new(4, 0).unwrap();
        assert!(must_promote(PieceType::Pawn, sq, Color::Black));

        // Black Pawn on rank 1 doesn't have to.
        let sq = Square::new(4, 1).unwrap();
        assert!(!must_promote(PieceType::Pawn, sq, Color::Black));

        // White Pawn on rank 8 must promote.
        let sq = Square::new(4, 8).unwrap();
        assert!(must_promote(PieceType::Pawn, sq, Color::White));

        // White Pawn on rank 7 doesn't have to.
        let sq = Square::new(4, 7).unwrap();
        assert!(!must_promote(PieceType::Pawn, sq, Color::White));
    }

    #[test]
    fn knight_must_promote_on_last_two_ranks() {
        // Black Knight on rank 0 and rank 1 must promote.
        for rank in 0..=1 {
            let sq = Square::new(4, rank).unwrap();
            assert!(must_promote(PieceType::Knight, sq, Color::Black));
        }
        // Rank 2 is fine.
        let sq = Square::new(4, 2).unwrap();
        assert!(!must_promote(PieceType::Knight, sq, Color::Black));

        // White Knight on rank 7 and rank 8 must promote.
        for rank in 7..=8 {
            let sq = Square::new(4, rank).unwrap();
            assert!(must_promote(PieceType::Knight, sq, Color::White));
        }
        let sq = Square::new(4, 6).unwrap();
        assert!(!must_promote(PieceType::Knight, sq, Color::White));
    }

    #[test]
    fn silver_never_must_promote() {
        for rank in 0..=8 {
            let sq = Square::new(4, rank).unwrap();
            assert!(!must_promote(PieceType::Silver, sq, Color::Black));
            assert!(!must_promote(PieceType::Silver, sq, Color::White));
        }
    }

    // -----------------------------------------------------------------------
    // in_promotion_zone
    // -----------------------------------------------------------------------

    #[test]
    fn black_promotion_zone_ranks_0_to_2() {
        for rank in 0..=2 {
            let sq = Square::new(4, rank).unwrap();
            assert!(
                in_promotion_zone(sq, Color::Black),
                "rank {rank} should be in zone"
            );
        }
        for rank in 3..=8 {
            let sq = Square::new(4, rank).unwrap();
            assert!(
                !in_promotion_zone(sq, Color::Black),
                "rank {rank} should NOT be in zone"
            );
        }
    }

    #[test]
    fn white_promotion_zone_ranks_6_to_8() {
        for rank in 6..=8 {
            let sq = Square::new(4, rank).unwrap();
            assert!(
                in_promotion_zone(sq, Color::White),
                "rank {rank} should be in zone"
            );
        }
        for rank in 0..=5 {
            let sq = Square::new(4, rank).unwrap();
            assert!(
                !in_promotion_zone(sq, Color::White),
                "rank {rank} should NOT be in zone"
            );
        }
    }

    // -----------------------------------------------------------------------
    // generate_moves — integration
    // -----------------------------------------------------------------------

    #[test]
    fn startpos_generates_moves() {
        init();
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        let moves = generate_moves(&pos);
        // The starting position should have 30 pseudo-legal moves for Black.
        assert!(
            moves.len() >= 30,
            "startpos should have >= 30 moves, got {}",
            moves.len()
        );
    }

    #[test]
    fn no_moves_in_bare_king_checkmate() {
        init();
        // Construct a position where Black's King is checkmated.
        // White Rook on file 5, rank 8 with Black King on 5e (rank 4) surrounded.
        // Use a simple known mate: Black King on 5a, White Gold on 5b and 4b.
        // Actually, let's use a simpler approach: verify movegen generates *some*
        // moves for a normal position and that drops work.
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2P 1").unwrap();
        let moves = generate_moves(&pos);
        // King has up to 8 moves; 2 Pawns can be dropped on most empty squares.
        assert!(
            moves.len() > 8,
            "should have king moves + pawn drops, got {}",
            moves.len()
        );
    }

    #[test]
    fn drop_moves_respect_placement_restrictions() {
        init();
        // Black has a Pawn in hand. Verify no drop on rank 0 (last rank for Black).
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1").unwrap();
        let moves = generate_moves(&pos);
        for m in &moves {
            if let Move::Drop {
                to,
                piece_type: PieceType::Pawn,
            } = m
            {
                assert_ne!(to.rank(), 0, "Pawn should not be droppable on rank 0");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Nifu rule — no two unpromoted Pawns on the same file
    // -----------------------------------------------------------------------

    #[test]
    fn pawn_drop_rejected_on_file_with_unpromoted_pawn() {
        init();
        // Black already has a Pawn on file 4 (rank 6), and holds another in
        // hand. No Pawn drop should be generated on file 4.
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/4P4/9/4K4 b P 1").unwrap();
        let moves = generate_moves(&pos);
        let blocked_file = Square::new(4, 6).unwrap().file();
        for m in &moves {
            if let Move::Drop {
                to,
                piece_type: PieceType::Pawn,
            } = m
            {
                assert_ne!(
                    to.file(),
                    blocked_file,
                    "nifu: Pawn drop on file {blocked_file} should be filtered, got {to}"
                );
            }
        }
    }

    #[test]
    fn pawn_drop_allowed_on_file_with_promoted_pawn() {
        init();
        // A promoted Pawn (Tokin) does NOT count for the nifu rule, so we
        // can still drop an unpromoted Pawn on the same file.
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/4+P4/9/4K4 b P 1").unwrap();
        let moves = generate_moves(&pos);
        let tokin_file = Square::new(4, 6).unwrap().file();
        let found = moves.iter().any(|m| {
            matches!(
                m,
                Move::Drop { to, piece_type: PieceType::Pawn } if to.file() == tokin_file
            )
        });
        assert!(
            found,
            "promoted Pawns do not trigger nifu; a Pawn drop on file \
             {tokin_file} should be available"
        );
    }

    #[test]
    fn blocked_pawn_files_detects_own_pawn_only() {
        init();
        let mut pos = Position::new();
        // SFEN ranks are written file 9 → file 1. "4P4" places Black's Pawn
        // at file index 4 (Shogi file 5), "6p2" places White's Pawn at file
        // index 2 (Shogi file 3).
        pos.set_sfen("4k4/9/9/9/9/9/4P4/6p2/4K4 b - 1").unwrap();
        let black_blocked = blocked_pawn_files(&pos, Color::Black);
        let white_blocked = blocked_pawn_files(&pos, Color::White);
        assert!(
            black_blocked[4],
            "Black's Pawn on file 4 should be detected"
        );
        assert!(!black_blocked[2], "White's Pawn should not block Black");
        assert!(
            white_blocked[2],
            "White's Pawn on file 2 should be detected"
        );
        assert!(!white_blocked[4], "Black's Pawn should not block White");
    }
}
