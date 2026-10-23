//! NNUE model loading and inference for Shogi (Phase 3).
//!
//! This module loads a pre-trained NNUE model exported from ShogiDojo's Python
//! training pipeline and evaluates positions using the learned weights.
//!
//! Two inference paths are provided:
//!
//!   * [`NnueModel::evaluate`] — stateless "full recompute" from an SFEN.
//!     Re-runs the Feature Transformer from scratch every call. Correct
//!     but slow. Used as a reference implementation in tests.
//!
//!   * [`NnueEvaluator`] — stateful evaluator with an incremental
//!     accumulator. Maintains the FT output across moves so that each
//!     `evaluate()` at a search leaf skips the FT recompute entirely. This
//!     is the "Efficiently Updatable" half of NNUE and is what the search
//!     uses in production.
//!
//! The evaluation pipeline (both paths):
//!   1. Parse the board position
//!   2. Compute HalfKAv2 sparse feature indices (matching Python's features.py)
//!   3. Feature Transformer: sum up weight vectors for active features
//!   4. Concatenate STM + NSTM perspectives → 512-dim
//!   5. ClippedReLU → L1 → ClippedReLU → L2 → ClippedReLU → L3 → raw score
//!   6. Scale to centipawns
//!
//! The feature extraction MUST match the Python implementation exactly,
//! otherwise the trained weights will produce garbage. Key reference files:
//!   - `src/shogi_dojo/nnue/constants.py` — dimensions, piece type IDs
//!   - `src/shogi_dojo/nnue/features.py` — index computation logic

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use shogi::{Color, Move, Piece, PieceType, Position, Square};

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
/// Weights are stored as flat `Vec<f32>` arrays and are read-only after
/// loading, so many search threads can share one [`Arc<NnueModel>`].
///
/// Per-thread mutable state (accumulator, snapshot stack) lives in
/// [`NnueEvaluator`], which borrows the model via `Arc`.
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

    /// Load a model from a binary file exported by `scripts/nnue/export_nnue.py`.
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

    /// Evaluate a position using a full FT recompute.
    ///
    /// This is the **reference implementation** — it rebuilds the Feature
    /// Transformer output from scratch on every call, which is correct but
    /// slow. Production search uses [`NnueEvaluator`] which keeps an
    /// incremental accumulator instead. Tests compare the two to catch
    /// drift.
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

// ============================================================================
// Incremental accumulator — the "Efficiently Updatable" half of NNUE
// ============================================================================

/// Per-perspective Feature Transformer output.
///
/// This is the core state that makes NNUE fast. Instead of recomputing the
/// FT sum from scratch every leaf, we keep the running sum here and patch
/// it when the search makes or unmakes a move. Only a handful of features
/// change per move (one piece leaves a square, one piece arrives, possibly
/// a capture and a hand update) — orders of magnitude less work than
/// re-summing ~40 features across two perspectives.
///
/// Two perspectives are kept:
///
/// * `black_acc`: the FT output from Black's perspective (board seen
///   directly, colors preserved, king bucket based on Black's king).
/// * `white_acc`: the FT output from White's perspective (board rotated
///   180°, colors swapped, king bucket based on White's king).
///
/// At evaluate-time we pick whichever corresponds to the side-to-move
/// first in the concatenated eval-net input.
#[derive(Clone)]
struct Accumulator {
    black_acc: Vec<f32>,
    white_acc: Vec<f32>,
    black_bucket: usize,
    white_bucket: usize,
}

impl Accumulator {
    fn new(ft_out_dim: usize) -> Self {
        Self {
            black_acc: vec![0.0; ft_out_dim],
            white_acc: vec![0.0; ft_out_dim],
            black_bucket: 0,
            white_bucket: 0,
        }
    }

    /// Add `weight[index]` into `acc`.
    fn add_feature(acc: &mut [f32], model: &NnueModel, index: usize) {
        let base = index * model.ft_out_dim;
        let w = &model.ft_weight[base..base + model.ft_out_dim];
        for (a, &v) in acc.iter_mut().zip(w.iter()) {
            *a += v;
        }
    }

    /// Subtract `weight[index]` from `acc`.
    fn sub_feature(acc: &mut [f32], model: &NnueModel, index: usize) {
        let base = index * model.ft_out_dim;
        let w = &model.ft_weight[base..base + model.ft_out_dim];
        for (a, &v) in acc.iter_mut().zip(w.iter()) {
            *a -= v;
        }
    }

    /// Rebuild both perspectives from scratch for `pos`. Called once before
    /// search starts, and whenever we can't safely patch (e.g. after a
    /// shogi-rs internal error or when we enter a fresh root position).
    #[allow(clippy::needless_range_loop)] // Indexed loops mirror the Python reference implementation
    fn refresh_from(&mut self, model: &NnueModel, pos: &Position) {
        let (pieces, _stm, hand, black_king_sq, white_king_sq) = extract_position(pos);

        self.black_bucket = king_square_to_bucket(mirror_file(black_king_sq));
        self.white_bucket = king_square_to_bucket(mirror_file(flip_square(white_king_sq)));

        self.black_acc.copy_from_slice(&model.ft_bias);
        self.white_acc.copy_from_slice(&model.ft_bias);

        // Board pieces
        for &(pt, color, sq) in &pieces {
            let (b_idx, w_idx) =
                board_feature_indices(self.black_bucket, self.white_bucket, pt, color, sq);
            Self::add_feature(&mut self.black_acc, model, b_idx);
            Self::add_feature(&mut self.white_acc, model, w_idx);
        }

        // Hand pieces (thermometer)
        for color in 0..NUM_COLORS {
            let mut offset = 0usize;
            for i in 0..HAND_PIECE_TYPES.len() {
                let count = hand[color][i] as usize;
                let max_count = HAND_PIECE_MAX[i];
                for slot in 0..count {
                    let (b_idx, w_idx) = hand_feature_indices(
                        self.black_bucket,
                        self.white_bucket,
                        color,
                        offset,
                        slot,
                    );
                    Self::add_feature(&mut self.black_acc, model, b_idx);
                    Self::add_feature(&mut self.white_acc, model, w_idx);
                }
                offset += max_count;
            }
        }
    }
}

/// Return the (Black-perspective, White-perspective) feature indices for a
/// non-king board piece at `py_sq`.
fn board_feature_indices(
    black_bucket: usize,
    white_bucket: usize,
    piece_type: usize,
    color: usize,
    py_sq: usize,
) -> (usize, usize) {
    let b_sq = mirror_file(py_sq);
    let b_idx =
        black_bucket * HALF_FEATURE_DIM + (piece_type * NUM_COLORS + color) * NUM_SQUARES + b_sq;
    let w_sq = mirror_file(flip_square(py_sq));
    let w_color = 1 - color;
    let w_idx =
        white_bucket * HALF_FEATURE_DIM + (piece_type * NUM_COLORS + w_color) * NUM_SQUARES + w_sq;
    (b_idx, w_idx)
}

/// Return the (Black-perspective, White-perspective) feature indices for a
/// single hand-piece thermometer slot. `holder_color` is the color that
/// holds the piece (0=Black, 1=White); `section_offset` is the cumulative
/// offset up to this piece-type's section (0 for Pawn, 18 for Lance, ...);
/// `slot` is the zero-indexed thermometer position (0..max_count).
fn hand_feature_indices(
    black_bucket: usize,
    white_bucket: usize,
    holder_color: usize,
    section_offset: usize,
    slot: usize,
) -> (usize, usize) {
    let b_idx = black_bucket * HALF_FEATURE_DIM
        + BOARD_FEATURE_DIM
        + holder_color * HAND_FEATURES_PER_SIDE
        + section_offset
        + slot;
    let w_color = 1 - holder_color;
    let w_idx = white_bucket * HALF_FEATURE_DIM
        + BOARD_FEATURE_DIM
        + w_color * HAND_FEATURES_PER_SIDE
        + section_offset
        + slot;
    (b_idx, w_idx)
}

/// Cumulative hand-section offset for `HAND_PIECE_TYPES[i]`.
fn hand_section_offset(hand_type_idx: usize) -> usize {
    HAND_PIECE_MAX.iter().take(hand_type_idx).sum()
}

/// Position of `pt` in `HAND_PIECE_TYPES`, or `None` if `pt` can't be in hand
/// (Kings, promoted pieces after they've been unpromoted are handled by the
/// caller — this expects an already-unpromoted base type).
fn hand_type_index(pt: PieceType) -> Option<usize> {
    HAND_PIECE_TYPES.iter().position(|&p| p == pt)
}

/// Unpromote a piece type to its base form (the form it takes in hand).
/// Non-promoted types are returned unchanged.
fn unpromote(pt: PieceType) -> PieceType {
    match pt {
        PieceType::ProPawn => PieceType::Pawn,
        PieceType::ProLance => PieceType::Lance,
        PieceType::ProKnight => PieceType::Knight,
        PieceType::ProSilver => PieceType::Silver,
        PieceType::ProBishop => PieceType::Bishop,
        PieceType::ProRook => PieceType::Rook,
        other => other,
    }
}

/// Promoted form of a promotable piece type. Panics on un-promotable types
/// (King, Gold) or types that are already promoted — the caller is
/// responsible for only calling this when `promote == true`.
fn promote(pt: PieceType) -> PieceType {
    match pt {
        PieceType::Pawn => PieceType::ProPawn,
        PieceType::Lance => PieceType::ProLance,
        PieceType::Knight => PieceType::ProKnight,
        PieceType::Silver => PieceType::ProSilver,
        PieceType::Bishop => PieceType::ProBishop,
        PieceType::Rook => PieceType::ProRook,
        other => panic!("promote() called on non-promotable type {:?}", other),
    }
}

// ============================================================================
// Stateful evaluator — the search-facing entry point
// ============================================================================

/// NNUE evaluator with an incremental accumulator.
///
/// Each search thread owns one. The `Arc<NnueModel>` holds the read-only
/// weights (shared across threads); the mutable state — the current
/// accumulator plus a stack of pre-move snapshots — lives here.
///
/// Lifecycle:
///
/// * On first `evaluate`, the accumulator is refreshed from the position.
///   Subsequent evaluates in the same search path reuse the live accumulator.
/// * `on_make` snapshots the current accumulator onto the undo stack, then
///   patches it forward for the move.
/// * `on_unmake` pops the snapshot back into the live accumulator — O(1)
///   in logical cost, a single `Vec` swap in practice.
/// * `refresh` is called when returning to the game root (e.g. between
///   searches at different real game positions) so the accumulator resyncs
///   to the new board state rather than drifting from cumulative edits.
pub struct NnueEvaluator {
    model: Arc<NnueModel>,
    acc: Accumulator,
    /// Snapshot stack. Uses a pool of pre-allocated `Accumulator` slots so
    /// that `on_make` is a `copy_from_slice` (constant memory bandwidth,
    /// no heap allocation). `stack_depth` is the logical top of the stack;
    /// the pool itself (`stack_pool.len()`) grows monotonically to the
    /// deepest recursion seen so far.
    stack_pool: Vec<Accumulator>,
    stack_depth: usize,
    /// Reusable scratch buffers for `eval_net`. Allocating three `Vec`s per
    /// leaf would dwarf the actual math; these grow to the right size on
    /// first use and then stay put.
    eval_x: Vec<f32>,
    eval_l1: Vec<f32>,
    eval_l2: Vec<f32>,
    initialized: bool,
}

impl NnueEvaluator {
    /// Build a fresh evaluator around a shared model.
    pub fn new(model: Arc<NnueModel>) -> Self {
        let ft_out_dim = model.ft_out_dim;
        let l1_dim = model.l1_dim;
        let l2_dim = model.l2_dim;
        Self {
            acc: Accumulator::new(ft_out_dim),
            stack_pool: Vec::with_capacity(64),
            stack_depth: 0,
            eval_x: vec![0.0; ft_out_dim * 2],
            eval_l1: vec![0.0; l1_dim],
            eval_l2: vec![0.0; l2_dim],
            model,
            initialized: false,
        }
    }

    /// Push the current accumulator onto the undo stack. Reuses an existing
    /// slot if the pool has capacity; otherwise allocates one fresh slot
    /// (amortized O(1) over the search).
    fn push_snapshot(&mut self) {
        let ft_out_dim = self.model.ft_out_dim;
        if self.stack_depth == self.stack_pool.len() {
            self.stack_pool.push(Accumulator::new(ft_out_dim));
        }
        let slot = &mut self.stack_pool[self.stack_depth];
        slot.black_acc.copy_from_slice(&self.acc.black_acc);
        slot.white_acc.copy_from_slice(&self.acc.white_acc);
        slot.black_bucket = self.acc.black_bucket;
        slot.white_bucket = self.acc.white_bucket;
        self.stack_depth += 1;
    }

    /// Restore the accumulator from the top of the undo stack.
    fn pop_snapshot(&mut self) {
        debug_assert!(
            self.stack_depth > 0,
            "pop_snapshot on empty stack (unbalanced on_make/on_unmake)"
        );
        self.stack_depth -= 1;
        let slot = &self.stack_pool[self.stack_depth];
        self.acc.black_acc.copy_from_slice(&slot.black_acc);
        self.acc.white_acc.copy_from_slice(&slot.white_acc);
        self.acc.black_bucket = slot.black_bucket;
        self.acc.white_bucket = slot.white_bucket;
    }

    /// Apply the forward accumulator update for a move that is about to be
    /// executed. Caller is responsible for invoking this *before*
    /// `pos.make_move(m)` so that `pos_before` is accurate.
    ///
    /// For king moves that change the bucket of the moving side, the
    /// moving side's accumulator is refreshed from the post-move state
    /// rather than patched. Non-king moves are pure add/subtract deltas.
    fn apply_make_forward(&mut self, pos_before: &Position, m: &Move) {
        let model = self.model.clone();
        match *m {
            Move::Normal {
                from,
                to,
                promote: promoted,
            } => {
                // If `from` has no piece, the move is illegal and the caller's
                // `make_move` will fail. The snapshot on the undo stack will
                // be popped by the matching `on_unmake`, restoring the valid
                // accumulator — so we can safely bail out instead of patching
                // with nonsense data.
                let Some(moving) = pos_before.piece_at(from) else {
                    return;
                };
                let victim = pos_before.piece_at(to);
                let from_py = square_to_python_index(from);
                let to_py = square_to_python_index(to);

                // --- Step 1: remove the captured piece, if any ---
                if let Some(v) = victim {
                    // Kings are never captured in a legal game; guard anyway.
                    if v.piece_type != PieceType::King {
                        let v_pt = piece_type_id(v.piece_type);
                        let v_color = color_id(v.color);
                        let (b_idx, w_idx) = board_feature_indices(
                            self.acc.black_bucket,
                            self.acc.white_bucket,
                            v_pt,
                            v_color,
                            to_py,
                        );
                        Accumulator::sub_feature(&mut self.acc.black_acc, &model, b_idx);
                        Accumulator::sub_feature(&mut self.acc.white_acc, &model, w_idx);

                        // The victim enters the mover's hand as its base
                        // (unpromoted) type. The thermometer slot to activate
                        // is the current hand count (pre-move), since we are
                        // about to add one more.
                        let base = unpromote(v.piece_type);
                        if let Some(h_idx) = hand_type_index(base) {
                            let old_count = pos_before.hand(Piece {
                                piece_type: base,
                                color: moving.color,
                            }) as usize;
                            let offset = hand_section_offset(h_idx);
                            let (b_idx, w_idx) = hand_feature_indices(
                                self.acc.black_bucket,
                                self.acc.white_bucket,
                                color_id(moving.color),
                                offset,
                                old_count,
                            );
                            Accumulator::add_feature(&mut self.acc.black_acc, &model, b_idx);
                            Accumulator::add_feature(&mut self.acc.white_acc, &model, w_idx);
                        }
                    }
                }

                // --- Step 2: remove the moving piece from `from` ---
                // Kings aren't features, so we only touch non-king pieces
                // here. The king's effect is purely via its bucket.
                if moving.piece_type != PieceType::King {
                    let m_pt = piece_type_id(moving.piece_type);
                    let m_color = color_id(moving.color);
                    let (b_idx, w_idx) = board_feature_indices(
                        self.acc.black_bucket,
                        self.acc.white_bucket,
                        m_pt,
                        m_color,
                        from_py,
                    );
                    Accumulator::sub_feature(&mut self.acc.black_acc, &model, b_idx);
                    Accumulator::sub_feature(&mut self.acc.white_acc, &model, w_idx);
                }

                // --- Step 3: handle king bucket change (if any) ---
                // A king move may put the king on a different rank, which
                // shifts every feature index for that perspective. In that
                // case we refresh the moving side's accumulator from the
                // post-move state; the opposite side's bucket is unchanged
                // so we simply continue patching it below.
                if moving.piece_type == PieceType::King {
                    let new_king_py = to_py;
                    let new_bucket = match moving.color {
                        Color::Black => king_square_to_bucket(mirror_file(new_king_py)),
                        Color::White => {
                            king_square_to_bucket(mirror_file(flip_square(new_king_py)))
                        }
                    };
                    let old_bucket = match moving.color {
                        Color::Black => self.acc.black_bucket,
                        Color::White => self.acc.white_bucket,
                    };
                    if new_bucket != old_bucket {
                        // Refresh just the mover's side from the post-move
                        // board state (derived from pos_before + m). We skip
                        // step 4 for this perspective because the refresh
                        // already accounts for the arriving piece, the
                        // vacated square, and any victim / hand change.
                        self.refresh_side_post_move(pos_before, m, moving.color);
                        // The opposite perspective still needs step 4's
                        // "add moving piece at `to`" if the moving piece is
                        // non-king — but kings aren't features, so nothing
                        // to add there. Done.
                        return;
                    }
                }

                // --- Step 4: add the moving piece at `to` (possibly
                // promoted). Skipped for kings (not a feature). ---
                if moving.piece_type != PieceType::King {
                    let new_pt = if promoted {
                        promote(moving.piece_type)
                    } else {
                        moving.piece_type
                    };
                    let m_pt = piece_type_id(new_pt);
                    let m_color = color_id(moving.color);
                    let (b_idx, w_idx) = board_feature_indices(
                        self.acc.black_bucket,
                        self.acc.white_bucket,
                        m_pt,
                        m_color,
                        to_py,
                    );
                    Accumulator::add_feature(&mut self.acc.black_acc, &model, b_idx);
                    Accumulator::add_feature(&mut self.acc.white_acc, &model, w_idx);
                }
            }

            Move::Drop { to, piece_type } => {
                // A drop: one hand thermometer slot goes away, one board
                // feature appears. No captures are possible in a drop.
                let mover = pos_before.side_to_move();
                let to_py = square_to_python_index(to);

                // Remove the topmost thermometer slot of the dropped type
                // from the mover's hand. Slot to remove = new count after
                // drop = old_count - 1.
                if let Some(h_idx) = hand_type_index(piece_type) {
                    let old_count = pos_before.hand(Piece {
                        piece_type,
                        color: mover,
                    }) as usize;
                    if old_count == 0 {
                        // Illegal drop (empty hand). `make_move` will reject
                        // it; the snapshot-restore on `on_unmake` undoes
                        // anything we touched below, so just skip the sub.
                        return;
                    }
                    let slot_to_remove = old_count - 1;
                    let offset = hand_section_offset(h_idx);
                    let (b_idx, w_idx) = hand_feature_indices(
                        self.acc.black_bucket,
                        self.acc.white_bucket,
                        color_id(mover),
                        offset,
                        slot_to_remove,
                    );
                    Accumulator::sub_feature(&mut self.acc.black_acc, &model, b_idx);
                    Accumulator::sub_feature(&mut self.acc.white_acc, &model, w_idx);
                }

                // Add the dropped piece to the board. Drops never deliver a
                // promoted piece (shogi rule) so `piece_type` is already the
                // base form.
                let m_pt = piece_type_id(piece_type);
                let m_color = color_id(mover);
                let (b_idx, w_idx) = board_feature_indices(
                    self.acc.black_bucket,
                    self.acc.white_bucket,
                    m_pt,
                    m_color,
                    to_py,
                );
                Accumulator::add_feature(&mut self.acc.black_acc, &model, b_idx);
                Accumulator::add_feature(&mut self.acc.white_acc, &model, w_idx);
            }
        }
    }

    /// Recompute the Black or White half of the accumulator from the
    /// logical post-move board/hand state, without cloning `Position`.
    /// Called only when a king move changes that side's bucket.
    ///
    /// The post-move state is derived by taking `pos_before`'s pieces and
    /// applying the move in-place to a small local view. This avoids a
    /// `Position::clone` (shogi-rs does not expose cheap cloning).
    #[allow(clippy::needless_range_loop)] // Indexed loops mirror the Python reference implementation
    fn refresh_side_post_move(&mut self, pos_before: &Position, m: &Move, side: Color) {
        let model = self.model.clone();
        let (mut pieces, _stm, mut hand, mut black_king, mut white_king) =
            extract_position(pos_before);

        match *m {
            Move::Normal {
                from,
                to,
                promote: promoted,
            } => {
                let moving = pos_before
                    .piece_at(from)
                    .expect("Normal move with no piece at `from`");
                let victim = pos_before.piece_at(to);
                let from_py = square_to_python_index(from);
                let to_py = square_to_python_index(to);

                // Remove victim from pieces + add to mover's hand.
                if let Some(v) = victim
                    && v.piece_type != PieceType::King
                {
                    let v_pt = piece_type_id(v.piece_type);
                    let v_color = color_id(v.color);
                    pieces.retain(|&(pt, col, sq)| !(pt == v_pt && col == v_color && sq == to_py));
                    let base = unpromote(v.piece_type);
                    if let Some(h_idx) = hand_type_index(base) {
                        hand[color_id(moving.color)][h_idx] += 1;
                    }
                }

                // Move the piece (king, in this branch's caller).
                if moving.piece_type == PieceType::King {
                    match moving.color {
                        Color::Black => black_king = to_py,
                        Color::White => white_king = to_py,
                    }
                } else {
                    // Non-king (shouldn't be reached from refresh_side_post_move's
                    // only caller, which gates on King, but handle for
                    // completeness / future use).
                    pieces.retain(|&(pt, col, sq)| {
                        !(pt == piece_type_id(moving.piece_type)
                            && col == color_id(moving.color)
                            && sq == from_py)
                    });
                    let new_pt = if promoted {
                        promote(moving.piece_type)
                    } else {
                        moving.piece_type
                    };
                    pieces.push((piece_type_id(new_pt), color_id(moving.color), to_py));
                }
            }
            Move::Drop { to, piece_type } => {
                let mover = pos_before.side_to_move();
                let to_py = square_to_python_index(to);
                if let Some(h_idx) = hand_type_index(piece_type) {
                    hand[color_id(mover)][h_idx] = hand[color_id(mover)][h_idx].saturating_sub(1);
                }
                pieces.push((piece_type_id(piece_type), color_id(mover), to_py));
            }
        }

        // Recompute the requested side's bucket and accumulator from the
        // post-move logical state. We leave the opposite side's accumulator
        // alone — its bucket did not change; the caller patches it via
        // the normal add/sub path.
        match side {
            Color::Black => {
                let new_bucket = king_square_to_bucket(mirror_file(black_king));
                self.acc.black_bucket = new_bucket;
                self.acc.black_acc.copy_from_slice(&model.ft_bias);
                for &(pt, color, sq) in &pieces {
                    let b_sq = mirror_file(sq);
                    let idx = new_bucket * HALF_FEATURE_DIM
                        + (pt * NUM_COLORS + color) * NUM_SQUARES
                        + b_sq;
                    Accumulator::add_feature(&mut self.acc.black_acc, &model, idx);
                }
                for color in 0..NUM_COLORS {
                    let mut offset = 0usize;
                    for i in 0..HAND_PIECE_TYPES.len() {
                        let count = hand[color][i] as usize;
                        let max_count = HAND_PIECE_MAX[i];
                        for slot in 0..count {
                            let idx = new_bucket * HALF_FEATURE_DIM
                                + BOARD_FEATURE_DIM
                                + color * HAND_FEATURES_PER_SIDE
                                + offset
                                + slot;
                            Accumulator::add_feature(&mut self.acc.black_acc, &model, idx);
                        }
                        offset += max_count;
                    }
                }

                // Also patch the opposite (White) side for victim removal &
                // hand gain, since step 3 returned early before step 4. The
                // caller already handled victim removal and hand add for the
                // opposite perspective in step 1. But our caller returns
                // before step 4, which for a king move is a no-op anyway
                // (king isn't a feature), so nothing more to do here.
            }
            Color::White => {
                let new_bucket = king_square_to_bucket(mirror_file(flip_square(white_king)));
                self.acc.white_bucket = new_bucket;
                self.acc.white_acc.copy_from_slice(&model.ft_bias);
                for &(pt, color, sq) in &pieces {
                    let w_sq = mirror_file(flip_square(sq));
                    let w_color = 1 - color;
                    let idx = new_bucket * HALF_FEATURE_DIM
                        + (pt * NUM_COLORS + w_color) * NUM_SQUARES
                        + w_sq;
                    Accumulator::add_feature(&mut self.acc.white_acc, &model, idx);
                }
                for color in 0..NUM_COLORS {
                    let mut offset = 0usize;
                    for i in 0..HAND_PIECE_TYPES.len() {
                        let count = hand[color][i] as usize;
                        let max_count = HAND_PIECE_MAX[i];
                        for slot in 0..count {
                            let w_color = 1 - color;
                            let idx = new_bucket * HALF_FEATURE_DIM
                                + BOARD_FEATURE_DIM
                                + w_color * HAND_FEATURES_PER_SIDE
                                + offset
                                + slot;
                            Accumulator::add_feature(&mut self.acc.white_acc, &model, idx);
                        }
                        offset += max_count;
                    }
                }
            }
        }
    }

    /// Run the eval-net half (ClippedReLU + L1 + L2 + L3) against the
    /// current accumulator, pick the perspective ordering from `stm`, and
    /// return a centipawn score. Reuses persistent scratch buffers to keep
    /// the hot path allocation-free.
    #[allow(clippy::needless_range_loop)] // Indexed loops mirror the Python reference implementation
    fn eval_net(&mut self, stm: Color) -> i32 {
        let model = self.model.clone();
        let (stm_acc, nstm_acc) = match stm {
            Color::Black => (&self.acc.black_acc, &self.acc.white_acc),
            Color::White => (&self.acc.white_acc, &self.acc.black_acc),
        };

        let concat_dim = model.ft_out_dim * 2;
        // Concat + ClippedReLU into the persistent scratch buffer.
        for (dst, src) in self
            .eval_x
            .iter_mut()
            .zip(stm_acc.iter().chain(nstm_acc.iter()))
        {
            *dst = src.clamp(0.0, 1.0);
        }

        // L1
        self.eval_l1.copy_from_slice(&model.l1_bias);
        for i in 0..model.l1_dim {
            let row_base = i * concat_dim;
            let mut acc = self.eval_l1[i];
            for j in 0..concat_dim {
                acc += model.l1_weight[row_base + j] * self.eval_x[j];
            }
            self.eval_l1[i] = acc.clamp(0.0, 1.0);
        }

        // L2
        self.eval_l2.copy_from_slice(&model.l2_bias);
        for i in 0..model.l2_dim {
            let row_base = i * model.l1_dim;
            let mut acc = self.eval_l2[i];
            for j in 0..model.l1_dim {
                acc += model.l2_weight[row_base + j] * self.eval_l1[j];
            }
            self.eval_l2[i] = acc.clamp(0.0, 1.0);
        }

        // L3
        let mut raw = model.l3_bias[0];
        for j in 0..model.l2_dim {
            raw += model.l3_weight[j] * self.eval_l2[j];
        }

        (raw * model.score_scaling) as i32
    }
}

impl Evaluator for NnueEvaluator {
    fn evaluate(&mut self, pos: &Position) -> i32 {
        if !self.initialized {
            self.acc.refresh_from(&self.model, pos);
            self.initialized = true;
        }
        self.eval_net(pos.side_to_move())
    }

    fn on_make(&mut self, pos_before: &Position, m: &Move) {
        if !self.initialized {
            self.acc.refresh_from(&self.model, pos_before);
            self.initialized = true;
        }
        self.push_snapshot();
        self.apply_make_forward(pos_before, m);
    }

    fn on_unmake(&mut self) {
        self.pop_snapshot();
    }

    fn refresh(&mut self, pos: &Position) {
        self.acc.refresh_from(&self.model, pos);
        self.stack_depth = 0;
        self.initialized = true;
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
    // `as_chunks::<4>()` splits the bytes into fixed-size `[u8; 4]` arrays
    // (plus a remainder, empty here since byte_len = count * 4), so each
    // chunk converts to an f32 without a fallible `try_into`.
    slice
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
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

    // -----------------------------------------------------------------------
    // Incremental accumulator — correctness vs. full recompute
    //
    // The strategy: build a small synthetic NnueModel with deterministic
    // weights, then assert that NnueEvaluator (which maintains an
    // incremental accumulator across make/unmake) returns the same score
    // as NnueModel::evaluate (full recompute) at every point in a move
    // sequence. Any divergence means the forward update or the snapshot
    // stack has a bug.
    //
    // We keep `ft_out_dim` small (8) and hand-craft reproducible weights
    // via a linear-congruential formula so the test is hermetic (no
    // model.nnue dependency).
    // -----------------------------------------------------------------------

    use shogi::Move;
    use std::sync::Arc;

    /// Build a synthetic NnueModel with deterministic weights for testing.
    /// Same dimensions as production (19,638 × ft_out_dim) so feature
    /// indices match Python exactly; smaller ft_out_dim keeps tests fast.
    fn synthetic_model(ft_out_dim: usize, l1_dim: usize, l2_dim: usize) -> NnueModel {
        let ft_input_dim = 9 * HALF_FEATURE_DIM; // 19638
        let mut seed: u32 = 0x1234_5678;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Map to a small symmetric range so activations stay in [0, 1]-ish
            // after summing ~40 features.
            (seed as i32 as f32) * 1e-11
        };

        let ft_weight: Vec<f32> = (0..ft_input_dim * ft_out_dim).map(|_| next()).collect();
        let ft_bias: Vec<f32> = (0..ft_out_dim).map(|_| next()).collect();

        let concat_dim = ft_out_dim * 2;
        let l1_weight: Vec<f32> = (0..l1_dim * concat_dim).map(|_| next() * 100.0).collect();
        let l1_bias: Vec<f32> = (0..l1_dim).map(|_| next() * 100.0).collect();
        let l2_weight: Vec<f32> = (0..l2_dim * l1_dim).map(|_| next() * 100.0).collect();
        let l2_bias: Vec<f32> = (0..l2_dim).map(|_| next() * 100.0).collect();
        let l3_weight: Vec<f32> = (0..l2_dim).map(|_| next() * 100.0).collect();
        let l3_bias: Vec<f32> = vec![next() * 100.0];

        NnueModel {
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
        }
    }

    fn assert_incremental_matches_full(
        eval: &mut NnueEvaluator,
        model: &NnueModel,
        pos: &Position,
        label: &str,
    ) {
        let incr = eval.evaluate(pos);
        let full = model.evaluate(pos);
        assert_eq!(
            incr, full,
            "{label}: incremental eval {incr} != full recompute {full}"
        );
    }

    #[test]
    fn evaluator_matches_full_on_startpos() {
        init();
        let model = Arc::new(synthetic_model(8, 4, 4));
        let mut eval = NnueEvaluator::new(Arc::clone(&model));
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        eval.refresh(&pos);
        assert_incremental_matches_full(&mut eval, &model, &pos, "startpos");
    }

    #[test]
    fn evaluator_matches_full_on_complex_midgame() {
        init();
        let model = Arc::new(synthetic_model(8, 4, 4));
        let mut eval = NnueEvaluator::new(Arc::clone(&model));
        // Position with captures in hand on both sides and a promoted piece.
        let mut pos = Position::new();
        pos.set_sfen(
            "ln1gk2nl/1r1s2gb1/p1+Pppp1pp/2p4P1/9/2P6/PP1PPPP1P/1B1S2R2/LN1GKGSNL w S2P 1",
        )
        .unwrap();
        eval.refresh(&pos);
        assert_incremental_matches_full(&mut eval, &model, &pos, "complex midgame");
    }

    #[test]
    fn evaluator_tracks_through_simple_move_sequence() {
        init();
        let model = Arc::new(synthetic_model(8, 4, 4));
        let mut eval = NnueEvaluator::new(Arc::clone(&model));
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        eval.refresh(&pos);
        assert_incremental_matches_full(&mut eval, &model, &pos, "before any move");

        // Play: 7g7f (Black pawn) — a plain forward move.
        let m = Move::Normal {
            from: Square::new(2, 6).unwrap(), // 7g (shogi-rs file 2 = Shogi file 7)
            to: Square::new(2, 5).unwrap(),   // 7f
            promote: false,
        };
        eval.on_make(&pos, &m);
        pos.make_move(m).unwrap();
        assert_incremental_matches_full(&mut eval, &model, &pos, "after 7g7f");

        // Unmake and verify we're back to startpos eval.
        pos.unmake_move().unwrap();
        eval.on_unmake();
        assert_incremental_matches_full(&mut eval, &model, &pos, "after unmake 7g7f");
    }

    #[test]
    fn evaluator_tracks_through_capture_and_hand_gain() {
        init();
        let model = Arc::new(synthetic_model(8, 4, 4));
        let mut eval = NnueEvaluator::new(Arc::clone(&model));
        // Position where Black's rook can capture an undefended White pawn
        // and gain a pawn in hand.
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/4p4/9/9/9/4R4/9/4K4 b - 1").unwrap();
        eval.refresh(&pos);
        assert_incremental_matches_full(&mut eval, &model, &pos, "before capture");

        // Rook captures pawn: 5g5c (file 4, rank 6 → file 4, rank 2).
        let m = Move::Normal {
            from: Square::new(4, 6).unwrap(),
            to: Square::new(4, 2).unwrap(),
            promote: false,
        };
        eval.on_make(&pos, &m);
        pos.make_move(m).unwrap();
        assert_incremental_matches_full(&mut eval, &model, &pos, "after capture");

        pos.unmake_move().unwrap();
        eval.on_unmake();
        assert_incremental_matches_full(&mut eval, &model, &pos, "after unmake capture");
    }

    #[test]
    fn evaluator_tracks_promotion() {
        init();
        let model = Arc::new(synthetic_model(8, 4, 4));
        let mut eval = NnueEvaluator::new(Arc::clone(&model));
        // Black pawn on 7c can step to 7b and promote.
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/2P6/9/9/9/9/9/4K4 b - 1").unwrap();
        eval.refresh(&pos);
        assert_incremental_matches_full(&mut eval, &model, &pos, "pre-promotion");

        // Pawn is at Shogi file 7, rank 3. shogi-rs file 0 = Shogi file 1,
        // so Shogi file 7 = shogi-rs file 6. Rank 3 = shogi-rs rank 2.
        // Moving to Shogi file 7, rank 2 (shogi-rs rank 1) promotes (rank ≤ 2).
        let m = Move::Normal {
            from: Square::new(6, 2).unwrap(),
            to: Square::new(6, 1).unwrap(),
            promote: true,
        };
        eval.on_make(&pos, &m);
        pos.make_move(m).unwrap();
        assert_incremental_matches_full(&mut eval, &model, &pos, "after promoting pawn");

        pos.unmake_move().unwrap();
        eval.on_unmake();
        assert_incremental_matches_full(&mut eval, &model, &pos, "after unmake promotion");
    }

    #[test]
    fn evaluator_tracks_drop() {
        init();
        let model = Arc::new(synthetic_model(8, 4, 4));
        let mut eval = NnueEvaluator::new(Arc::clone(&model));
        // Black has a Pawn in hand and an empty central square to drop on.
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1").unwrap();
        eval.refresh(&pos);
        assert_incremental_matches_full(&mut eval, &model, &pos, "before drop");

        let m = Move::Drop {
            to: Square::new(4, 4).unwrap(), // center
            piece_type: PieceType::Pawn,
        };
        eval.on_make(&pos, &m);
        pos.make_move(m).unwrap();
        assert_incremental_matches_full(&mut eval, &model, &pos, "after drop");

        pos.unmake_move().unwrap();
        eval.on_unmake();
        assert_incremental_matches_full(&mut eval, &model, &pos, "after unmake drop");
    }

    #[test]
    fn evaluator_tracks_king_move_without_bucket_change() {
        init();
        let model = Arc::new(synthetic_model(8, 4, 4));
        let mut eval = NnueEvaluator::new(Arc::clone(&model));
        // Black king shuffles on its own back rank (stays in bucket 8).
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/9/3K5 b - 1").unwrap();
        eval.refresh(&pos);
        assert_incremental_matches_full(&mut eval, &model, &pos, "before king shuffle");

        // Black king: file 5 rank 8 → file 4 rank 8 (same rank = same bucket).
        let m = Move::Normal {
            from: Square::new(5, 8).unwrap(),
            to: Square::new(4, 8).unwrap(),
            promote: false,
        };
        eval.on_make(&pos, &m);
        pos.make_move(m).unwrap();
        assert_incremental_matches_full(&mut eval, &model, &pos, "after king same-rank move");

        pos.unmake_move().unwrap();
        eval.on_unmake();
        assert_incremental_matches_full(&mut eval, &model, &pos, "after unmake king shuffle");
    }

    #[test]
    fn evaluator_tracks_king_move_that_changes_bucket() {
        init();
        let model = Arc::new(synthetic_model(8, 4, 4));
        let mut eval = NnueEvaluator::new(Arc::clone(&model));
        // Black king advances from rank 8 to rank 7 — different bucket.
        let mut pos = Position::new();
        pos.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").unwrap();
        eval.refresh(&pos);
        assert_incremental_matches_full(&mut eval, &model, &pos, "before bucket-crossing king");

        let m = Move::Normal {
            from: Square::new(4, 8).unwrap(), // rank 8
            to: Square::new(4, 7).unwrap(),   // rank 7 — different bucket
            promote: false,
        };
        eval.on_make(&pos, &m);
        pos.make_move(m).unwrap();
        assert_incremental_matches_full(
            &mut eval,
            &model,
            &pos,
            "after king move that crosses bucket",
        );

        pos.unmake_move().unwrap();
        eval.on_unmake();
        assert_incremental_matches_full(
            &mut eval,
            &model,
            &pos,
            "after unmake bucket-crossing king",
        );
    }

    #[test]
    fn evaluator_tracks_failed_make_rollback() {
        init();
        let model = Arc::new(synthetic_model(8, 4, 4));
        let mut eval = NnueEvaluator::new(Arc::clone(&model));
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        eval.refresh(&pos);
        let pre = eval.evaluate(&pos);

        // Attempt a clearly illegal move (pawn drop on a file that already
        // has an unpromoted pawn). `on_make` is called speculatively; then
        // make_move fails; then `on_unmake` rolls back.
        let illegal = Move::Drop {
            to: Square::new(0, 4).unwrap(),
            piece_type: PieceType::Pawn,
        };
        eval.on_make(&pos, &illegal);
        // Depending on hand state, make_move may fail. For this position
        // Black has no pawn in hand, so make_move must fail — but our
        // accumulator was already patched. Roll back.
        assert!(pos.make_move(illegal).is_err());
        eval.on_unmake();

        let post = eval.evaluate(&pos);
        assert_eq!(pre, post, "rollback of failed make should restore eval");
    }

    #[test]
    fn evaluator_tracks_through_random_legal_sequence() {
        // Play a few random legal ply from startpos using PST-chosen moves,
        // asserting at every position that incremental == full.
        init();
        let model = Arc::new(synthetic_model(8, 4, 4));
        let mut eval = NnueEvaluator::new(Arc::clone(&model));
        let mut pos = Position::new();
        pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        eval.refresh(&pos);

        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        use rand::seq::SliceRandom;
        let mut rng = SmallRng::seed_from_u64(0xF00D_BEEF);

        for ply in 0..20 {
            // Find a legal move (non-exhaustive; just the first we can make).
            let pseudo = crate::movegen::generate_moves(&pos);
            let mut ordered: Vec<Move> = pseudo.into_iter().collect();
            ordered.shuffle(&mut rng);
            let mut played = false;
            for m in ordered {
                eval.on_make(&pos, &m);
                if pos.make_move(m).is_err() {
                    eval.on_unmake();
                    continue;
                }
                played = true;
                assert_incremental_matches_full(
                    &mut eval,
                    &model,
                    &pos,
                    &format!("after ply {ply}"),
                );
                break;
            }
            if !played {
                // No legal moves — game over; nothing more to assert.
                break;
            }
        }
    }
}
