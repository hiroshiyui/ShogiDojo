"""Tests for shogi_dojo.alphazero.encoding — board and move encoding.

Coverage goals:
  - Constants are internally consistent (POLICY_DIM = 9·9·139, channel
    offsets line up).
  - Position encoding produces the expected shape, places pieces on the
    correct planes, applies the side-to-move perspective flip, and
    normalizes the scalar planes within their documented caps.
  - Move encoding is a faithful round-trip for every kind of move
    (queen-like sliding, knight, drops, promoting variants), under both
    Black and White perspective.
  - Invalid inputs raise informative ValueErrors rather than silently
    producing wrong indices.
  - The legal-move mask has the right shape and is True at exactly the
    encoded indices.
"""

from __future__ import annotations

import pytest
import torch

from shogi_dojo.alphazero.encoding import (
    ALL_PIECE_TYPES_ORDERED,
    NUM_DROP_PLANES,
    NUM_INPUT_CHANNELS,
    NUM_KNIGHT_OFFSETS,
    NUM_MOVE_PLANES,
    NUM_QUEEN_DIRECTIONS,
    NUM_QUEEN_DISTANCES,
    NUM_QUEEN_PLANES,
    NUM_SLIDING_PLANES,
    PLY_CAP,
    POLICY_DIM,
    REPETITION_CAP,
    Move,
    Position,
    decode_move,
    encode_move,
    encode_position,
    legal_move_mask,
    position_from_sfen,
)
from shogi_dojo.nnue.constants import (
    BISHOP,
    BLACK,
    GOLD,
    HAND_PIECE_TYPES,
    KING,
    KNIGHT,
    LANCE,
    NUM_FILES,
    NUM_RANKS,
    NUM_SQUARES,
    PAWN,
    ROOK,
    SILVER,
    WHITE,
)


# Standard Shogi starting position SFEN (Black to move, no pieces in hand).
STARTPOS = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def _empty_hand() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Two-side hand structure with all counts zero."""
    zeros = tuple(0 for _ in HAND_PIECE_TYPES)
    return (zeros, zeros)


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


class TestEncodingConstants:
    """The constants should add up the way the docstring claims."""

    def test_policy_dim_is_9x9x139(self):
        assert POLICY_DIM == NUM_RANKS * NUM_FILES * NUM_MOVE_PLANES
        assert POLICY_DIM == 9 * 9 * 139
        assert POLICY_DIM == 11_259

    def test_move_plane_count_decomposition(self):
        assert NUM_QUEEN_PLANES == NUM_QUEEN_DIRECTIONS * NUM_QUEEN_DISTANCES
        assert NUM_QUEEN_PLANES == 64
        assert NUM_KNIGHT_OFFSETS == 2
        assert NUM_SLIDING_PLANES == NUM_QUEEN_PLANES + NUM_KNIGHT_OFFSETS == 66
        assert NUM_DROP_PLANES == len(HAND_PIECE_TYPES) == 7
        # 64 queen + 2 knight + 64 promoting queen + 2 promoting knight + 7 drops
        assert NUM_MOVE_PLANES == 64 + 2 + 64 + 2 + 7 == 139

    def test_input_channel_count(self):
        # 14 own + 14 opp + 7 own hand + 7 opp hand + rep + stm + ply
        assert NUM_INPUT_CHANNELS == 14 + 14 + 7 + 7 + 1 + 1 + 1
        assert NUM_INPUT_CHANNELS == 45

    def test_all_piece_types_ordered_includes_king(self):
        # The encoder uses 14 piece-type planes (NNUE only has 13 because
        # it handles King via king buckets); make sure King is in the list.
        assert KING in ALL_PIECE_TYPES_ORDERED
        assert len(ALL_PIECE_TYPES_ORDERED) == 14


# ---------------------------------------------------------------------------
# encode_position — basic shape and scalar planes
# ---------------------------------------------------------------------------


class TestEncodePositionShape:
    def test_returns_float32_tensor_of_expected_shape(self):
        pos = Position(
            pieces=(),
            side_to_move=BLACK,
            hand=_empty_hand(),
        )
        tensor = encode_position(pos)
        assert tensor.shape == (NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
        assert tensor.dtype == torch.float32

    def test_empty_position_has_only_constant_planes_set(self):
        # No pieces, no hand. The piece and hand planes should be all-zero;
        # the rep/stm/ply scalar planes should be set per the position.
        pos = Position(
            pieces=(),
            side_to_move=BLACK,
            hand=_empty_hand(),
            move_number=1,
            repetition_count=0,
        )
        tensor = encode_position(pos)
        # Channels 0..41 (piece + hand planes) should all be zero.
        assert torch.all(tensor[:42] == 0.0)
        # Channel 42 (rep) is 0 because repetition_count=0.
        assert torch.all(tensor[42] == 0.0)
        # Channel 43 (STM) is always 1.0.
        assert torch.all(tensor[43] == 1.0)
        # Channel 44 (ply) is 1/PLY_CAP because move_number=1.
        assert torch.allclose(tensor[44], torch.full((9, 9), 1.0 / PLY_CAP))


class TestEncodePositionScalarPlanes:
    def test_repetition_plane_normalized_and_capped(self):
        for rep in [0, 1, 2, 3, 5, 100]:
            pos = Position(
                pieces=(), side_to_move=BLACK, hand=_empty_hand(),
                repetition_count=rep,
            )
            tensor = encode_position(pos)
            expected = min(rep, REPETITION_CAP) / REPETITION_CAP
            assert torch.allclose(tensor[42], torch.full((9, 9), expected)), \
                f"rep={rep} expected plane fill {expected}"

    def test_ply_plane_normalized_and_capped(self):
        for ply in [1, 10, 100, PLY_CAP, PLY_CAP * 2]:
            pos = Position(
                pieces=(), side_to_move=BLACK, hand=_empty_hand(),
                move_number=ply,
            )
            tensor = encode_position(pos)
            expected = min(ply, PLY_CAP) / PLY_CAP
            assert torch.allclose(tensor[44], torch.full((9, 9), expected))

    def test_stm_plane_always_one_after_perspective_flip(self):
        # The STM plane should be 1.0 regardless of which color is moving,
        # since the encoder always presents the position from STM's view.
        for stm in [BLACK, WHITE]:
            pos = Position(
                pieces=(), side_to_move=stm, hand=_empty_hand(),
            )
            tensor = encode_position(pos)
            assert torch.all(tensor[43] == 1.0)


# ---------------------------------------------------------------------------
# encode_position — pieces and hand
# ---------------------------------------------------------------------------


class TestEncodePositionPieces:
    def test_single_own_piece_appears_on_correct_plane(self):
        # Black pawn on square 76 (rank 8 file 4 — center file, near bottom).
        # With STM=Black, no flip, so the STM frame y=8, x=4.
        # Pawn is plane index 0 in ALL_PIECE_TYPES_ORDERED, so own-pawn = plane 0.
        pos = Position(
            pieces=((PAWN, BLACK, 76),),
            side_to_move=BLACK,
            hand=_empty_hand(),
        )
        tensor = encode_position(pos)
        assert tensor[0, 8, 4] == 1.0
        # The rest of the own-pawn plane should be zero.
        own_pawn_plane = tensor[0]
        zeros_count = (own_pawn_plane == 0.0).sum().item()
        assert zeros_count == NUM_SQUARES - 1

    def test_single_opponent_piece_appears_on_opp_plane(self):
        # Black is STM; a White pawn on square 4 (rank 0 file 4) is the
        # opponent. Opponent-pawn plane = 14 (NUM_OWN_PIECE_PLANES).
        pos = Position(
            pieces=((PAWN, WHITE, 4),),
            side_to_move=BLACK,
            hand=_empty_hand(),
        )
        tensor = encode_position(pos)
        assert tensor[14, 0, 4] == 1.0
        # And nothing on the own-pawn plane.
        assert torch.all(tensor[0] == 0.0)

    def test_perspective_flip_swaps_owner_and_rotates_square(self):
        # With STM=White, a White piece becomes "own", and its square is
        # rotated 180°. A White pawn on raw square 76 (rank 8 file 4)
        # rotates to (rank 0 file 4) in STM coordinates.
        pos = Position(
            pieces=((PAWN, WHITE, 76),),
            side_to_move=WHITE,
            hand=_empty_hand(),
        )
        tensor = encode_position(pos)
        assert tensor[0, 0, 4] == 1.0  # own-pawn at flipped position
        # Nothing on the opponent-pawn plane.
        assert torch.all(tensor[14] == 0.0)

    def test_piece_type_ordering_matches_constants_order(self):
        # Place one piece of each type and verify it lands on the plane
        # index dictated by ALL_PIECE_TYPES_ORDERED.
        for plane_offset, piece_type in enumerate(ALL_PIECE_TYPES_ORDERED):
            pos = Position(
                pieces=((piece_type, BLACK, 40),),  # center square
                side_to_move=BLACK,
                hand=_empty_hand(),
            )
            tensor = encode_position(pos)
            assert tensor[plane_offset, 4, 4] == 1.0, \
                f"piece type {piece_type} expected on plane {plane_offset}"


class TestEncodePositionHand:
    def test_own_hand_count_normalized_by_max(self):
        # Black holds 9 pawns. Pawn max is 18, so the plane should be
        # filled with 9/18 = 0.5.
        hand = (
            (9, 0, 0, 0, 0, 0, 0),  # Black: 9 pawns
            (0, 0, 0, 0, 0, 0, 0),  # White: empty
        )
        pos = Position(pieces=(), side_to_move=BLACK, hand=hand)
        tensor = encode_position(pos)
        # Own-pawn-hand plane = 28.
        assert torch.allclose(tensor[28], torch.full((9, 9), 0.5))

    def test_opponent_hand_appears_on_opp_hand_planes(self):
        # White holds 2 rooks (max 2 → 1.0). With STM=Black, White is opp.
        hand = (
            (0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 2),  # White: 2 rooks
        )
        pos = Position(pieces=(), side_to_move=BLACK, hand=hand)
        tensor = encode_position(pos)
        # Opp-hand-rook plane = 35 (opp base) + 6 (rook idx) = 41.
        assert torch.allclose(tensor[41], torch.full((9, 9), 1.0))

    def test_perspective_flip_swaps_hand_owner(self):
        # With STM=White, the White hand becomes "own".
        hand = (
            (0, 0, 0, 0, 0, 0, 0),
            (18, 0, 0, 0, 0, 0, 0),  # White: 18 pawns
        )
        pos = Position(pieces=(), side_to_move=WHITE, hand=hand)
        tensor = encode_position(pos)
        # Own-pawn-hand plane = 28; should be 18/18 = 1.0.
        assert torch.allclose(tensor[28], torch.full((9, 9), 1.0))


# ---------------------------------------------------------------------------
# position_from_sfen
# ---------------------------------------------------------------------------


class TestPositionFromSfen:
    def test_starting_position_has_two_kings_and_38_pieces(self):
        pos = position_from_sfen(STARTPOS)
        # 40 board pieces total (20 per side, including kings).
        assert len(pos.pieces) == 40
        kings = [p for p in pos.pieces if p[0] == KING]
        assert len(kings) == 2

    def test_starting_position_side_to_move_is_black(self):
        pos = position_from_sfen(STARTPOS)
        assert pos.side_to_move == BLACK

    def test_starting_position_hands_are_empty(self):
        pos = position_from_sfen(STARTPOS)
        assert all(c == 0 for c in pos.hand[BLACK])
        assert all(c == 0 for c in pos.hand[WHITE])

    def test_starting_position_encodes_to_valid_tensor(self):
        pos = position_from_sfen(STARTPOS)
        tensor = encode_position(pos)
        assert tensor.shape == (NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
        # 40 pieces total → 40 ones in the per-piece planes.
        per_piece = tensor[:28]   # own + opp piece planes
        assert per_piece.sum().item() == 40

    def test_starting_position_repetition_is_zero(self):
        pos = position_from_sfen(STARTPOS)
        assert pos.repetition_count == 0


# ---------------------------------------------------------------------------
# Move round-trip
# ---------------------------------------------------------------------------


def _round_trip(move: Move, side_to_move: int) -> Move:
    """Encode then decode a move; helper for round-trip tests."""
    idx = encode_move(move, side_to_move)
    return decode_move(idx, side_to_move)


class TestQueenMoveRoundTrip:
    @pytest.mark.parametrize("side_to_move", [BLACK, WHITE])
    @pytest.mark.parametrize(
        "from_sq,to_sq",
        [
            # Pawn one square forward (Black: from rank 6 to rank 5)
            (60, 51),
            # Lance straight forward (Black: from rank 8 file 0 → rank 0 file 0)
            (72, 0),
            # Bishop diagonal (from corner to corner)
            (0, 80),
            # Rook horizontal (rank 4 file 0 → rank 4 file 8)
            (36, 44),
            # King one step (e.g., rank 8 file 4 → rank 7 file 4)
            (76, 67),
            # Single-step backward (S direction)
            (40, 49),
        ],
    )
    def test_round_trip_preserves_move(self, from_sq, to_sq, side_to_move):
        move = Move(from_square=from_sq, to_square=to_sq)
        assert _round_trip(move, side_to_move) == move

    @pytest.mark.parametrize("side_to_move", [BLACK, WHITE])
    def test_round_trip_with_promotion(self, side_to_move):
        # Pawn promoting: from (rank 3 file 0) to (rank 2 file 0).
        move = Move(from_square=27, to_square=18, promote=True)
        assert _round_trip(move, side_to_move) == move


class TestKnightMoveRoundTrip:
    """Round-trip coverage for the knight planes.

    Note: knight moves are perspective-asymmetric. Shogi knights only
    move forward, so a move that is a legal knight jump in Black's frame
    becomes a backward (illegal) knight in White's frame. Each test uses
    a side-appropriate move rather than parametrizing one move over both
    colors.
    """

    def test_knight_forward_left_black(self):
        # Black knight at (rank 8 file 1) → (rank 6 file 0).
        # Black's STM frame: delta = (-2, -1) = forward-left (knight offset 0).
        move = Move(from_square=73, to_square=54)
        assert _round_trip(move, BLACK) == move

    def test_knight_forward_left_white(self):
        # White knight at (rank 0 file 7) → (rank 2 file 8) in raw coords.
        # White's STM frame after flip: delta = (-2, -1) = forward-left.
        move = Move(from_square=7, to_square=26)
        assert _round_trip(move, WHITE) == move

    def test_knight_forward_right_black(self):
        # Black knight at (rank 8 file 7) → (rank 6 file 8).
        # Black's STM frame: delta = (-2, +1) = forward-right (knight offset 1).
        move = Move(from_square=79, to_square=62)
        assert _round_trip(move, BLACK) == move

    def test_knight_forward_right_white(self):
        # White knight at (rank 0 file 1) → (rank 2 file 0) in raw coords.
        # White's STM frame after flip: delta = (-2, +1) = forward-right.
        move = Move(from_square=1, to_square=18)
        assert _round_trip(move, WHITE) == move

    def test_promoting_knight_round_trip_black(self):
        # Black promoting knight: from (rank 4 file 1) → (rank 2 file 0),
        # promote=True. STM-frame delta = (-2, -1) = forward-left.
        move = Move(from_square=37, to_square=18, promote=True)
        assert _round_trip(move, BLACK) == move

    def test_promoting_knight_round_trip_white(self):
        # White promoting knight: from (rank 4 file 7) → (rank 6 file 8),
        # promote=True. After flip: delta = (-2, -1) = forward-left.
        move = Move(from_square=43, to_square=62, promote=True)
        assert _round_trip(move, WHITE) == move


class TestDropMoveRoundTrip:
    @pytest.mark.parametrize("side_to_move", [BLACK, WHITE])
    @pytest.mark.parametrize(
        "drop_piece", [PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK],
    )
    def test_drop_round_trip(self, drop_piece, side_to_move):
        # Drop on a center square.
        move = Move(from_square=None, to_square=40, drop_piece=drop_piece)
        assert _round_trip(move, side_to_move) == move

    @pytest.mark.parametrize("side_to_move", [BLACK, WHITE])
    def test_drop_at_corners_round_trips(self, side_to_move):
        for to_sq in [0, 8, 72, 80]:
            move = Move(from_square=None, to_square=to_sq, drop_piece=PAWN)
            assert _round_trip(move, side_to_move) == move


# ---------------------------------------------------------------------------
# Move encoding error handling
# ---------------------------------------------------------------------------


class TestEncodeMoveErrors:
    def test_drop_without_drop_piece_raises(self):
        move = Move(from_square=None, to_square=40, drop_piece=None)
        with pytest.raises(ValueError, match="drop_piece"):
            encode_move(move, BLACK)

    def test_drop_with_promotion_raises(self):
        move = Move(from_square=None, to_square=40, drop_piece=PAWN, promote=True)
        with pytest.raises(ValueError, match="cannot promote"):
            encode_move(move, BLACK)

    def test_drop_of_non_droppable_piece_raises(self):
        # KING is not a droppable hand piece type.
        move = Move(from_square=None, to_square=40, drop_piece=KING)
        with pytest.raises(ValueError, match="HAND_PIECE_TYPES"):
            encode_move(move, BLACK)

    def test_unrepresentable_geometry_raises(self):
        # A "move" from (0,0) to (3, 5) is neither queen nor knight.
        move = Move(from_square=0, to_square=32)
        with pytest.raises(ValueError, match="not representable"):
            encode_move(move, BLACK)


class TestDecodeMoveErrors:
    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="outside"):
            decode_move(-1, BLACK)

    def test_index_too_large_raises(self):
        with pytest.raises(ValueError, match="outside"):
            decode_move(POLICY_DIM, BLACK)

    def test_decode_walking_off_board_raises(self):
        # A queen-N-distance-8 move from rank 0 walks off the board (you'd
        # land at rank -8). encode_move would never produce this index for
        # a legal Shogi move, but decode should raise rather than silently
        # produce garbage.
        # Plane for queen N distance 8 = direction 0, distance 8 → plane 7.
        # Source square (rank 0 file 0) → square index 0.
        bad_index = 7 * NUM_SQUARES + 0
        with pytest.raises(ValueError, match="off the board"):
            decode_move(bad_index, BLACK)


# ---------------------------------------------------------------------------
# legal_move_mask
# ---------------------------------------------------------------------------


class TestLegalMoveMask:
    def test_shape_and_dtype(self):
        mask = legal_move_mask([], BLACK)
        assert mask.shape == (NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)
        assert mask.dtype == torch.bool

    def test_empty_legal_list_produces_all_false(self):
        mask = legal_move_mask([], BLACK)
        assert not mask.any()

    def test_mask_active_at_encoded_indices(self):
        moves = [
            Move(from_square=60, to_square=51),                     # pawn forward
            Move(from_square=None, to_square=40, drop_piece=PAWN),  # drop
            Move(from_square=27, to_square=18, promote=True),       # promoting
        ]
        mask = legal_move_mask(moves, BLACK)

        # Verify the mask's True count matches the number of distinct moves.
        assert int(mask.sum().item()) == len(moves)

        # Verify each move's encoded position is True.
        for move in moves:
            idx = encode_move(move, BLACK)
            plane = idx // NUM_SQUARES
            sq = idx % NUM_SQUARES
            y, x = sq // NUM_FILES, sq % NUM_FILES
            assert mask[plane, y, x]


# ---------------------------------------------------------------------------
# Cross-checks: encoding two perspectives produces consistent geometry
# ---------------------------------------------------------------------------


class TestPerspectiveConsistency:
    def test_pawn_one_step_forward_uses_same_plane_either_color(self):
        # The "queen N distance 1" plane (plane 0) should be selected for
        # *either* a Black pawn one square forward or a White pawn one
        # square forward, because both look like "forward" in the STM
        # frame.
        black_move = Move(from_square=60, to_square=51)   # Black pawn N1
        white_move = Move(from_square=11, to_square=20)   # White pawn N1 in raw coords (= S in raw frame, N in STM frame)

        black_idx = encode_move(black_move, BLACK)
        white_idx = encode_move(white_move, WHITE)

        # Both indices must live on plane 0 (queen N distance 1).
        assert black_idx // NUM_SQUARES == 0
        assert white_idx // NUM_SQUARES == 0

    def test_knight_planes_used_for_either_color(self):
        # A Black knight forward-right and a White knight that looks the
        # same in STM frame should both land on plane 65 (knight FR).
        black_knight = Move(from_square=79, to_square=62)
        # In raw coords, White's "forward-right" knight at the symmetric
        # square is from (rank 0 file 1) to (rank 2 file 0).
        white_knight = Move(from_square=1, to_square=18)

        assert encode_move(black_knight, BLACK) // NUM_SQUARES == 65
        assert encode_move(white_knight, WHITE) // NUM_SQUARES == 65
