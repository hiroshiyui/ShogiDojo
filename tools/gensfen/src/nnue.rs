//! NNUE model loading and inference for Shogi (Phase 3).
//!
//! This module loads a pre-trained NNUE model exported from ShogiDojo's Python
//! training pipeline and evaluates positions using the learned weights.
//!
//! The evaluation pipeline:
//!   1. Parse the board position
//!   2. Compute HalfKAv2 sparse feature indices (matching Python's features.py)
//!   3. Feature Transformer: sum up weight vectors for active features
//!   4. Concatenate STM + NSTM perspectives → 512-dim
//!   5. ClippedReLU → L1 → ClippedReLU → L2 → ClippedReLU → L3 → raw score
//!   6. Scale to centipawns
//!
//! The feature extraction MUST match the Python implementation exactly,
//! otherwise the trained weights will produce garbage. Key reference files:
//!   - `src/shogi_dojo/constants.py` — dimensions, piece type IDs
//!   - `src/shogi_dojo/features.py` — index computation logic

use std::fs::File;
use std::io::Read;
use std::path::Path;

use shogi::{Color, Piece, PieceType, Position, Square};

use crate::eval::Evaluator;

// ============================================================================
// Constants (must match Python's constants.py exactly)
// ============================================================================

const NUM_RANKS: usize = 9;
const NUM_FILES: usize = 9;
const NUM_SQUARES: usize = 81;
const NUM_COLORS: usize = 2;
const NUM_PIECE_TYPES: usize = 13; // Excluding King

// HalfKAv2 feature dimensions
const BOARD_FEATURE_DIM: usize = NUM_PIECE_TYPES * NUM_COLORS * NUM_SQUARES; // 2106
const HAND_FEATURES_PER_SIDE: usize = 38; // 18+4+4+4+4+2+2
const HALF_FEATURE_DIM: usize = BOARD_FEATURE_DIM + HAND_FEATURES_PER_SIDE * NUM_COLORS; // 2182
const _NUM_KING_BUCKETS: usize = 9;

/// Python piece type IDs (must match constants.py).
/// These map shogi-rs PieceType → Python integer ID.
fn piece_type_id(pt: PieceType) -> usize {
    match pt {
        PieceType::Pawn => 0,
        PieceType::Lance => 1,
        PieceType::Knight => 2,
        PieceType::Silver => 3,
        PieceType::Gold => 4,
        PieceType::Bishop => 5,
        PieceType::Rook => 6,
        PieceType::ProPawn => 7,
        PieceType::ProLance => 8,
        PieceType::ProKnight => 9,
        PieceType::ProSilver => 10,
        PieceType::ProBishop => 11, // Horse
        PieceType::ProRook => 12,   // Dragon
        PieceType::King => panic!("King should not be passed to piece_type_id"),
    }
}

/// Color as integer (must match Python: BLACK=0, WHITE=1).
fn color_id(c: Color) -> usize {
    match c {
        Color::Black => 0,
        Color::White => 1,
    }
}

/// Hand piece types in the order used by Python's HAND_PIECE_TYPES.
const HAND_PIECE_TYPES: [PieceType; 7] = [
    PieceType::Pawn,
    PieceType::Lance,
    PieceType::Knight,
    PieceType::Silver,
    PieceType::Gold,
    PieceType::Bishop,
    PieceType::Rook,
];

/// Maximum count for each hand piece type (must match Python's HAND_PIECE_MAX).
const HAND_PIECE_MAX: [usize; 7] = [18, 4, 4, 4, 4, 2, 2];

// ============================================================================
// Square mapping between shogi-rs and Python
// ============================================================================

/// Convert shogi-rs Square to Python's square index (0-80).
///
/// Python indexes squares as: index = rank * 9 + (8 - file)
/// This scans left-to-right (file 9 → file 1), top-to-bottom (rank 1 → rank 9).
///
/// shogi-rs Square: file() is 0-8 where file 0 = Shogi "file 1" (rightmost),
///                  rank() is 0-8 where rank 0 = top row.
fn square_to_python_index(sq: Square) -> usize {
    let rank = sq.rank() as usize;
    let file = sq.file() as usize;
    // Python: file 0 in the index = Shogi file 9 (leftmost) = shogi-rs file 8
    // So: python_file_index = 8 - file
    rank * 9 + (8 - file)
}

/// Flip a square 180 degrees (rotate board for opponent's perspective).
/// Python: _flip_square(sq) → (8-rank)*9 + (8-file)
fn flip_square(sq_idx: usize) -> usize {
    let rank = sq_idx / NUM_FILES;
    let file = sq_idx % NUM_FILES;
    (NUM_RANKS - 1 - rank) * NUM_FILES + (NUM_FILES - 1 - file)
}

/// Mirror a square horizontally for left-right symmetry.
/// Python: mirror_file(sq) → if file > 4, mirror to 8-file
fn mirror_file(sq_idx: usize) -> usize {
    let rank = sq_idx / NUM_FILES;
    let mut file = sq_idx % NUM_FILES;
    if file > NUM_FILES / 2 {
        file = NUM_FILES - 1 - file;
    }
    rank * NUM_FILES + file
}

/// Map a king square to a bucket index (0-8) based on rank.
fn king_square_to_bucket(sq_idx: usize) -> usize {
    sq_idx / NUM_FILES
}

// ============================================================================
// NNUE Model
// ============================================================================

/// Binary format magic number: "NNUE" in little-endian ASCII
const MAGIC: u32 = 0x4E4E5545;

/// NNUE model loaded from a binary file.
///
/// The model contains all weight matrices and biases needed for inference.
/// Weights are stored as flat Vec<f32> arrays.
pub struct NnueModel {
    // Dimensions
    ft_input_dim: usize,
    ft_out_dim: usize,
    l1_dim: usize,
    l2_dim: usize,

    // Feature Transformer: weight is (ft_input_dim × ft_out_dim), bias is (ft_out_dim)
    ft_weight: Vec<f32>,
    ft_bias: Vec<f32>,

    // Linear layers: weight is (out × in), bias is (out)
    l1_weight: Vec<f32>,
    l1_bias: Vec<f32>,
    l2_weight: Vec<f32>,
    l2_bias: Vec<f32>,
    l3_weight: Vec<f32>,
    l3_bias: Vec<f32>,

    /// Score scaling factor (default 600.0, matching Python's training config).
    /// The network outputs a value in roughly [-1, 1] after sigmoid; we multiply
    /// by this to get centipawns.
    pub score_scaling: f32,
}

impl NnueModel {
    pub fn ft_input_dim(&self) -> usize {
        self.ft_input_dim
    }
    pub fn ft_out_dim(&self) -> usize {
        self.ft_out_dim
    }

    /// Load a model from a binary file exported by `scripts/export_nnue.py`.
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut file =
            File::open(path).map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        let mut cursor = 0usize;

        // Read header
        let magic = read_u32(&data, &mut cursor);
        if magic != MAGIC {
            return Err(format!(
                "Invalid magic number: expected 0x{:08X}, got 0x{:08X}",
                MAGIC, magic
            ));
        }
        let version = read_u32(&data, &mut cursor);
        if version != 1 {
            return Err(format!("Unsupported version: {}", version));
        }

        let ft_input_dim = read_u32(&data, &mut cursor) as usize;
        let ft_out_dim = read_u32(&data, &mut cursor) as usize;
        let l1_dim = read_u32(&data, &mut cursor) as usize;
        let l2_dim = read_u32(&data, &mut cursor) as usize;

        // Read weights
        let ft_weight = read_f32_vec(&data, &mut cursor, ft_input_dim * ft_out_dim);
        let ft_bias = read_f32_vec(&data, &mut cursor, ft_out_dim);
        let l1_weight = read_f32_vec(&data, &mut cursor, l1_dim * ft_out_dim * 2);
        let l1_bias = read_f32_vec(&data, &mut cursor, l1_dim);
        let l2_weight = read_f32_vec(&data, &mut cursor, l2_dim * l1_dim);
        let l2_bias = read_f32_vec(&data, &mut cursor, l2_dim);
        let l3_weight = read_f32_vec(&data, &mut cursor, l2_dim);
        let l3_bias = read_f32_vec(&data, &mut cursor, 1);

        if cursor != data.len() {
            return Err(format!(
                "Unexpected data: read {} bytes but file has {}",
                cursor,
                data.len()
            ));
        }

        Ok(Self {
            ft_input_dim,
            ft_out_dim,
            l1_dim,
            l2_dim,
            ft_weight,
            ft_bias,
            l1_weight,
            l1_bias,
            l2_weight,
            l2_bias,
            l3_weight,
            l3_bias,
            score_scaling: 600.0,
        })
    }

    /// Evaluate a position using the NNUE model.
    ///
    /// Returns a score in centipawns from the side-to-move's perspective.
    #[allow(clippy::needless_range_loop)] // Index loops match the Python reference implementation
    pub fn evaluate(&self, pos: &Position) -> i32 {
        // Step 1: Extract board state
        let (pieces, side_to_move, hand, black_king_sq, white_king_sq) = extract_position(pos);

        // Step 2: Compute HalfKAv2 feature indices
        let (stm_indices, nstm_indices) =
            make_feature_indices(&pieces, side_to_move, &hand, black_king_sq, white_king_sq);

        // Step 3: Feature Transformer — sparse lookup + sum
        let mut stm_out = self.ft_bias.clone();
        for &idx in &stm_indices {
            let base = idx * self.ft_out_dim;
            for j in 0..self.ft_out_dim {
                stm_out[j] += self.ft_weight[base + j];
            }
        }

        let mut nstm_out = self.ft_bias.clone();
        for &idx in &nstm_indices {
            let base = idx * self.ft_out_dim;
            for j in 0..self.ft_out_dim {
                nstm_out[j] += self.ft_weight[base + j];
            }
        }

        // Step 4: Concatenate [STM || NSTM] and apply ClippedReLU
        let concat_dim = self.ft_out_dim * 2;
        let mut x = Vec::with_capacity(concat_dim);
        for v in stm_out.iter().chain(nstm_out.iter()) {
            x.push(v.clamp(0.0, 1.0)); // ClippedReLU
        }

        // Step 5: L1 → ClippedReLU
        let mut l1_out = self.l1_bias.clone();
        for i in 0..self.l1_dim {
            for j in 0..concat_dim {
                // PyTorch Linear weight layout: weight[out_idx][in_idx]
                l1_out[i] += self.l1_weight[i * concat_dim + j] * x[j];
            }
            l1_out[i] = l1_out[i].clamp(0.0, 1.0);
        }

        // Step 6: L2 → ClippedReLU
        let mut l2_out = self.l2_bias.clone();
        for i in 0..self.l2_dim {
            for j in 0..self.l1_dim {
                l2_out[i] += self.l2_weight[i * self.l1_dim + j] * l1_out[j];
            }
            l2_out[i] = l2_out[i].clamp(0.0, 1.0);
        }

        // Step 7: L3 → raw score (no activation)
        let mut raw_score = self.l3_bias[0];
        for j in 0..self.l2_dim {
            raw_score += self.l3_weight[j] * l2_out[j];
        }

        // Step 8: Scale to centipawns
        // The network output is roughly in the range that sigmoid maps to [0,1].
        // Multiply by score_scaling to get centipawns.
        (raw_score * self.score_scaling) as i32
    }
}

impl Evaluator for NnueModel {
    fn evaluate(&self, pos: &Position) -> i32 {
        NnueModel::evaluate(self, pos)
    }
}

// ============================================================================
// Feature extraction (must match Python's features.py exactly)
// ============================================================================

/// Piece on the board: (piece_type_id, color_id, python_square_index)
type BoardPiece = (usize, usize, usize);

/// Hand piece counts: hand[color_id][hand_type_index] = count
type Hand = [[u8; 7]; 2];

/// Extract board state from a shogi-rs Position into the format needed
/// for HalfKAv2 feature computation.
fn extract_position(pos: &Position) -> (Vec<BoardPiece>, usize, Hand, usize, usize) {
    let mut pieces = Vec::new();
    let mut black_king_sq = 0usize;
    let mut white_king_sq = 0usize;

    // Scan all 81 squares
    for i in 0..81u8 {
        let sq = Square::from_index(i).unwrap();
        if let Some(piece) = pos.piece_at(sq) {
            let py_sq = square_to_python_index(sq);
            if piece.piece_type == PieceType::King {
                match piece.color {
                    Color::Black => black_king_sq = py_sq,
                    Color::White => white_king_sq = py_sq,
                }
            } else {
                pieces.push((
                    piece_type_id(piece.piece_type),
                    color_id(piece.color),
                    py_sq,
                ));
            }
        }
    }

    // Hand pieces
    let mut hand: Hand = [[0; 7]; 2];
    for (i, &pt) in HAND_PIECE_TYPES.iter().enumerate() {
        for &color in &[Color::Black, Color::White] {
            let piece = Piece {
                piece_type: pt,
                color,
            };
            hand[color_id(color)][i] = pos.hand(piece);
        }
    }

    let stm = color_id(pos.side_to_move());
    (pieces, stm, hand, black_king_sq, white_king_sq)
}

/// Compute HalfKAv2 feature indices for a position.
///
/// Returns (stm_indices, nstm_indices) — two lists of active feature indices.
/// This is a direct port of Python's `make_feature_indices()` in features.py.
#[allow(clippy::needless_range_loop)] // Index loops match the Python reference implementation
fn make_feature_indices(
    pieces: &[BoardPiece],
    side_to_move: usize,
    hand: &Hand,
    black_king_sq: usize,
    white_king_sq: usize,
) -> (Vec<usize>, Vec<usize>) {
    // King buckets for each perspective
    let black_bucket = king_square_to_bucket(mirror_file(black_king_sq));
    let white_bucket = king_square_to_bucket(mirror_file(flip_square(white_king_sq)));

    let mut black_indices = Vec::new();
    let mut white_indices = Vec::new();

    // Board piece features
    for &(piece_type, color, sq) in pieces {
        // Black's perspective
        let b_sq = mirror_file(sq);
        let b_color = color;
        let b_idx = black_bucket * HALF_FEATURE_DIM
            + (piece_type * NUM_COLORS + b_color) * NUM_SQUARES
            + b_sq;
        black_indices.push(b_idx);

        // White's perspective (flip board 180°, swap colors)
        let w_sq = mirror_file(flip_square(sq));
        let w_color = 1 - color;
        let w_idx = white_bucket * HALF_FEATURE_DIM
            + (piece_type * NUM_COLORS + w_color) * NUM_SQUARES
            + w_sq;
        white_indices.push(w_idx);
    }

    // Hand piece features (thermometer encoding)
    let hand_base = BOARD_FEATURE_DIM;
    for color in 0..NUM_COLORS {
        let mut offset = 0usize;
        for (i, &_pt) in HAND_PIECE_TYPES.iter().enumerate() {
            let count = hand[color][i] as usize;
            let max_count = HAND_PIECE_MAX[i];
            for c in 0..count {
                // Black perspective
                let b_color = color;
                let b_hand_idx = black_bucket * HALF_FEATURE_DIM
                    + hand_base
                    + b_color * HAND_FEATURES_PER_SIDE
                    + offset
                    + c;
                black_indices.push(b_hand_idx);

                // White perspective (swap color)
                let w_color = 1 - color;
                let w_hand_idx = white_bucket * HALF_FEATURE_DIM
                    + hand_base
                    + w_color * HAND_FEATURES_PER_SIDE
                    + offset
                    + c;
                white_indices.push(w_hand_idx);
            }
            offset += max_count;
        }
    }

    // Return (stm_indices, nstm_indices)
    if side_to_move == 0 {
        // BLACK to move
        (black_indices, white_indices)
    } else {
        (white_indices, black_indices)
    }
}

// ============================================================================
// Binary reading helpers
// ============================================================================

fn read_u32(data: &[u8], cursor: &mut usize) -> u32 {
    let val = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    val
}

fn read_f32_vec(data: &[u8], cursor: &mut usize, count: usize) -> Vec<f32> {
    let byte_len = count * 4;
    let slice = &data[*cursor..*cursor + byte_len];
    *cursor += byte_len;
    slice
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi::bitboard::Factory as BBFactory;

    fn init() {
        BBFactory::init();
    }

    // -----------------------------------------------------------------------
    // square_to_python_index
    // -----------------------------------------------------------------------

    #[test]
    fn python_index_corners() {
        init();
        // Top-left in Python is file 9 = shogi-rs file 8, rank 0 → index 0
        let sq = Square::new(8, 0).unwrap();
        assert_eq!(square_to_python_index(sq), 0);

        // Top-right in Python is file 1 = shogi-rs file 0, rank 0 → index 8
        let sq = Square::new(0, 0).unwrap();
        assert_eq!(square_to_python_index(sq), 8);

        // Bottom-left: shogi-rs file 8, rank 8 → index 72
        let sq = Square::new(8, 8).unwrap();
        assert_eq!(square_to_python_index(sq), 72);

        // Bottom-right: shogi-rs file 0, rank 8 → index 80
        let sq = Square::new(0, 8).unwrap();
        assert_eq!(square_to_python_index(sq), 80);
    }

    #[test]
    fn python_index_center() {
        init();
        // Center square: file 4, rank 4 → index 4*9 + (8-4) = 40
        let sq = Square::new(4, 4).unwrap();
        assert_eq!(square_to_python_index(sq), 40);
    }

    #[test]
    fn python_index_range() {
        init();
        for i in 0..81u8 {
            let sq = Square::from_index(i).unwrap();
            let py_idx = square_to_python_index(sq);
            assert!(py_idx < 81, "index out of range: {py_idx}");
        }
    }

    #[test]
    fn python_index_all_unique() {
        init();
        let mut seen = std::collections::HashSet::new();
        for i in 0..81u8 {
            let sq = Square::from_index(i).unwrap();
            let py_idx = square_to_python_index(sq);
            assert!(seen.insert(py_idx), "duplicate python index: {py_idx}");
        }
    }

    // -----------------------------------------------------------------------
    // flip_square
    // -----------------------------------------------------------------------

    #[test]
    fn flip_square_corners() {
        assert_eq!(flip_square(0), 80);
        assert_eq!(flip_square(80), 0);
        assert_eq!(flip_square(8), 72);
        assert_eq!(flip_square(72), 8);
    }

    #[test]
    fn flip_square_center_unchanged() {
        assert_eq!(flip_square(40), 40);
    }

    #[test]
    fn flip_square_double_identity() {
        for sq in 0..81 {
            assert_eq!(flip_square(flip_square(sq)), sq);
        }
    }

    // -----------------------------------------------------------------------
    // mirror_file
    // -----------------------------------------------------------------------

    #[test]
    fn mirror_file_left_half_unchanged() {
        // Files 0-4 (left half) should be unchanged.
        for rank in 0..9 {
            for file in 0..=4 {
                let sq = rank * 9 + file;
                assert_eq!(mirror_file(sq), sq, "sq={sq} should be unchanged");
            }
        }
    }

    #[test]
    fn mirror_file_right_half_mirrors() {
        for rank in 0..9 {
            for file in 5..9 {
                let sq = rank * 9 + file;
                let mirrored = mirror_file(sq);
                let expected_file = 8 - file;
                let expected = rank * 9 + expected_file;
                assert_eq!(mirrored, expected, "sq={sq} mirrored wrong");
            }
        }
    }

    // -----------------------------------------------------------------------
    // king_square_to_bucket
    // -----------------------------------------------------------------------

    #[test]
    fn bucket_equals_rank() {
        for sq in 0..81 {
            let expected_rank = sq / 9;
            assert_eq!(king_square_to_bucket(sq), expected_rank);
        }
    }

    // -----------------------------------------------------------------------
    // piece_type_id
    // -----------------------------------------------------------------------

    #[test]
    fn piece_type_ids_match_python() {
        assert_eq!(piece_type_id(PieceType::Pawn), 0);
        assert_eq!(piece_type_id(PieceType::Lance), 1);
        assert_eq!(piece_type_id(PieceType::Knight), 2);
        assert_eq!(piece_type_id(PieceType::Silver), 3);
        assert_eq!(piece_type_id(PieceType::Gold), 4);
        assert_eq!(piece_type_id(PieceType::Bishop), 5);
        assert_eq!(piece_type_id(PieceType::Rook), 6);
        assert_eq!(piece_type_id(PieceType::ProPawn), 7);
        assert_eq!(piece_type_id(PieceType::ProLance), 8);
        assert_eq!(piece_type_id(PieceType::ProKnight), 9);
        assert_eq!(piece_type_id(PieceType::ProSilver), 10);
        assert_eq!(piece_type_id(PieceType::ProBishop), 11);
        assert_eq!(piece_type_id(PieceType::ProRook), 12);
    }

    #[test]
    #[should_panic(expected = "King should not be passed")]
    fn piece_type_id_panics_on_king() {
        piece_type_id(PieceType::King);
    }

    // -----------------------------------------------------------------------
    // color_id
    // -----------------------------------------------------------------------

    #[test]
    fn color_ids_match_python() {
        assert_eq!(color_id(Color::Black), 0);
        assert_eq!(color_id(Color::White), 1);
    }

    // -----------------------------------------------------------------------
    // Constants match Python
    // -----------------------------------------------------------------------

    #[test]
    fn dimension_constants_match_python() {
        assert_eq!(BOARD_FEATURE_DIM, 13 * 2 * 81); // 2106
        assert_eq!(HAND_FEATURES_PER_SIDE, 38);
        assert_eq!(HALF_FEATURE_DIM, 2106 + 38 * 2); // 2182
    }

    // -----------------------------------------------------------------------
    // extract_position — integration
    // -----------------------------------------------------------------------

    #[test]
    fn extract_startpos_piece_count() {
        init();
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        let (pieces, stm, hand, bk, wk) = extract_position(&pos);
        // 40 pieces total, 2 kings excluded → 38 non-king pieces.
        assert_eq!(pieces.len(), 38);
        assert_eq!(stm, 0); // BLACK
        // No hand pieces.
        for color in 0..2 {
            for i in 0..7 {
                assert_eq!(hand[color][i], 0);
            }
        }
        // Kings should be on their starting squares.
        // Black King: rank 8, file 4 → python index = 8*9 + (8-4) = 76
        assert_eq!(bk, 76);
        // White King: rank 0, file 4 → python index = 0*9 + (8-4) = 4
        assert_eq!(wk, 4);
    }

    #[test]
    fn extract_position_with_hand_pieces() {
        init();
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2PBg 1").unwrap();
        let (_, stm, hand, _, _) = extract_position(&pos);
        assert_eq!(stm, 0);
        // Black: 2 Pawns (index 0), 1 Bishop (index 5)
        assert_eq!(hand[0][0], 2); // Pawn
        assert_eq!(hand[0][5], 1); // Bishop
        // White: 1 Gold (index 4)
        assert_eq!(hand[1][4], 1);
    }

    // -----------------------------------------------------------------------
    // make_feature_indices — feature index bounds
    // -----------------------------------------------------------------------

    #[test]
    fn feature_indices_in_bounds_startpos() {
        init();
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        let (pieces, stm, hand, bk, wk) = extract_position(&pos);
        let (stm_idx, nstm_idx) = make_feature_indices(&pieces, stm, &hand, bk, wk);

        let max_dim = _NUM_KING_BUCKETS * HALF_FEATURE_DIM; // 19638
        for &idx in stm_idx.iter().chain(nstm_idx.iter()) {
            assert!(
                idx < max_dim,
                "feature index {idx} out of bounds (max {max_dim})"
            );
        }
    }

    #[test]
    fn feature_indices_nonempty() {
        init();
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        let (pieces, stm, hand, bk, wk) = extract_position(&pos);
        let (stm_idx, nstm_idx) = make_feature_indices(&pieces, stm, &hand, bk, wk);
        assert!(!stm_idx.is_empty());
        assert!(!nstm_idx.is_empty());
        // 38 board pieces → at least 38 features per perspective.
        assert!(stm_idx.len() >= 38);
        assert!(nstm_idx.len() >= 38);
    }

    #[test]
    fn stm_swap_flips_perspectives() {
        init();
        let mut pos_b = Position::new();
        pos_b
            .set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        let mut pos_w = Position::new();
        pos_w
            .set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1")
            .unwrap();

        let (p_b, _, h_b, bk_b, wk_b) = extract_position(&pos_b);
        let (p_w, _, h_w, bk_w, wk_w) = extract_position(&pos_w);

        let (stm_b, nstm_b) = make_feature_indices(&p_b, 0, &h_b, bk_b, wk_b);
        let (stm_w, nstm_w) = make_feature_indices(&p_w, 1, &h_w, bk_w, wk_w);

        // Flipping side-to-move should swap STM ↔ NSTM.
        let mut sorted_stm_b = stm_b.clone();
        sorted_stm_b.sort();
        let mut sorted_nstm_w = nstm_w.clone();
        sorted_nstm_w.sort();
        assert_eq!(sorted_stm_b, sorted_nstm_w);

        let mut sorted_nstm_b = nstm_b.clone();
        sorted_nstm_b.sort();
        let mut sorted_stm_w = stm_w.clone();
        sorted_stm_w.sort();
        assert_eq!(sorted_nstm_b, sorted_stm_w);
    }

    // -----------------------------------------------------------------------
    // Binary reading helpers
    // -----------------------------------------------------------------------

    #[test]
    fn read_u32_little_endian() {
        let data = [0x42, 0x00, 0x00, 0x00];
        let mut cursor = 0;
        assert_eq!(read_u32(&data, &mut cursor), 0x42);
        assert_eq!(cursor, 4);
    }

    #[test]
    fn read_f32_vec_values() {
        // Encode two f32 values: 1.0 and -2.5
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&(-2.5f32).to_le_bytes());
        let mut cursor = 0;
        let result = read_f32_vec(&data, &mut cursor, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], -2.5);
        assert_eq!(cursor, 8);
    }

    // -----------------------------------------------------------------------
    // NnueModel::load — error cases
    // -----------------------------------------------------------------------

    #[test]
    fn load_nonexistent_file_errors() {
        let result = NnueModel::load(std::path::Path::new("/nonexistent/model.nnue"));
        assert!(result.is_err());
    }

    #[test]
    fn load_invalid_magic_errors() {
        let dir = std::env::temp_dir();
        let path = dir.join("bad_magic.nnue");
        // Write wrong magic number.
        let mut data = Vec::new();
        data.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
        std::fs::write(&path, &data).unwrap();
        let result = NnueModel::load(&path);
        assert!(result.is_err());
        let err = match result {
            Err(e) => e,
            Ok(_) => unreachable!(),
        };
        assert!(
            err.contains("Invalid magic"),
            "error should mention invalid magic: {err}"
        );
        std::fs::remove_file(&path).ok();
    }
}
