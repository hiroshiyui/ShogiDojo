"""Self-play game driver and replay buffer for AlphaZero training.

This module wires together the pieces built in earlier phases —
:mod:`encoding`, :mod:`network`, :mod:`mcts`, :mod:`rules` — into the
self-play half of an AlphaZero training loop.

The high-level picture (see
``doc/alphazero/AlphaZero_Training_Reference.md`` §5):

    for each iteration:
        for each self-play game:
            examples = play_game(network, ...)    ← this module
            replay_buffer.add_many(examples)      ← this module
        for each training step:
            batch = replay_buffer.sample(batch_size)   ← this module
            loss = alphazero_loss(network(batch[...]), ...)
            optimizer.step(loss)

A training example is ``(input_tensor, policy_target, value_target,
legal_mask)``. We pre-encode at self-play time and store tensors
directly in the buffer — simpler sampling and constant-cost per-batch
training at the cost of a couple of kilobytes per example. At the
survey's scale (~10–100K examples in the buffer) this is trivially
affordable.

The ``play_game`` function follows the standard AlphaZero self-play
recipe: for every move, run MCTS with Dirichlet noise at the root,
record the state plus MCTS visit distribution π at τ=1 (regardless of
what τ is used for the actual move), sample the move with τ=1 for the
first ``temperature_threshold`` plies and argmax afterward, apply the
move, and repeat. When the game terminates, backfill the outcome
``z`` into every recorded example from that position's STM
perspective.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import shogi
import torch

from shogi_dojo.alphazero.encoding import (
    NUM_MOVE_PLANES,
    encode_move,
    encode_position,
    legal_move_mask,
)
from shogi_dojo.alphazero.mcts import (
    MCTS,
    policy_target,
    sample_move,
)
from shogi_dojo.alphazero.rules import (
    DEFAULT_MOVE_LIMIT,
    apply_fn,
    board_to_position,
    is_terminal_fn,
    legal_moves_fn,
    make_evaluate_fn,
    shogi_move_to_move,
)
from shogi_dojo.nnue.constants import (
    BLACK,
    NUM_FILES,
    NUM_RANKS,
    NUM_SQUARES,
    WHITE,
)


# ============================================================================
# Training example
# ============================================================================


@dataclass(frozen=True)
class TrainingExample:
    """One ``(state, π, z, legal_mask)`` tuple produced by self-play.

    Attributes:
        input_tensor: The 45-channel board encoding,
            shape ``(NUM_INPUT_CHANNELS, 9, 9)``. Produced from the
            board's side-to-move perspective, so the trainer can feed
            the tensor to the network without additional transforms.
        policy_target: The MCTS visit distribution π at τ=1, packed
            into the policy head's output shape
            ``(NUM_MOVE_PLANES, 9, 9) == (139, 9, 9)``. Illegal
            entries are zero. Non-zero entries sum to 1 over legal
            moves.
        value_target: Scalar ``z ∈ {-1, 0, +1}`` — the eventual game
            outcome from this position's side-to-move's perspective.
            Filled in after the game terminates, not at the time the
            example is created.
        legal_mask: Boolean mask of shape ``(NUM_MOVE_PLANES, 9, 9)``
            marking which policy-output positions correspond to legal
            moves at this position. Used by the trainer's loss
            function to mask illegal logits before log-softmax.
    """

    input_tensor: torch.Tensor
    policy_target: torch.Tensor
    value_target: float
    legal_mask: torch.Tensor


# ============================================================================
# Helpers used by both play_game and the tests
# ============================================================================


def _policy_dict_to_tensor(
    visit_dict: dict[shogi.Move, float],
    side_to_move: int,
) -> torch.Tensor:
    """Pack the MCTS-derived π dict into a (139, 9, 9) tensor.

    The tensor has the same layout as the network's policy head output,
    so the trainer can compute cross-entropy directly without
    additional reshaping.
    """
    out = torch.zeros(NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES, dtype=torch.float32)
    for shogi_move, prob in visit_dict.items():
        our_move = shogi_move_to_move(shogi_move)
        idx = encode_move(our_move, side_to_move)
        plane = idx // NUM_SQUARES
        sq = idx % NUM_SQUARES
        y = sq // NUM_FILES
        x = sq % NUM_FILES
        out[plane, y, x] = prob
    return out


def _backfill_z(
    per_move_stms: list[int],
    terminal_stm: int,
    terminal_value: float,
) -> list[float]:
    """Compute the ``z`` target for every position in the game.

    AlphaZero records the eventual game outcome ``z`` at every
    position, from *that position's* side-to-move's perspective. For
    a two-player zero-sum game, the outcome for a player at ply ``t``
    is either equal to or the negation of the outcome for the player
    who moves at the terminal ply.

    Args:
        per_move_stms: The side-to-move at each recorded position, in
            game order.
        terminal_stm: The side-to-move at the terminal board (whoever
            would have moved next, had the game continued).
        terminal_value: The outcome from ``terminal_stm``'s
            perspective (``-1`` loss, ``0`` draw, ``+1`` win).

    Returns:
        A list of ``z`` values, same length as ``per_move_stms``.
    """
    return [
        terminal_value if stm == terminal_stm else -terminal_value
        for stm in per_move_stms
    ]


# ============================================================================
# One self-play game
# ============================================================================


def play_game(
    network: torch.nn.Module,
    num_simulations: int = 100,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.15,
    dirichlet_epsilon: float = 0.25,
    temperature_threshold: int = 30,
    move_limit: int = DEFAULT_MOVE_LIMIT,
    starting_board: Optional[shogi.Board] = None,
    device: torch.device | str = "cpu",
    rng: Optional[np.random.Generator] = None,
) -> list[TrainingExample]:
    """Run one full self-play game and return its training examples.

    MCTS drives every move. The network is queried via
    :func:`make_evaluate_fn` (one forward pass per leaf expansion,
    under ``torch.no_grad``).

    Hyperparameter defaults are the AlphaZero paper's values for Shogi
    where applicable, scaled down for survey-sized compute:

    - ``num_simulations=100`` — much less than the paper's 800, enough
      to demonstrate the algorithm without blowing out game wall-time.
    - ``c_puct=1.5`` — reasonable middle-ground; see
      ``AlphaZero_Training_Reference.md`` §4.
    - ``dirichlet_alpha=0.15, dirichlet_epsilon=0.25`` — Shogi values
      from the paper.
    - ``temperature_threshold=30`` — paper default. For moves at
      ``board.move_number <= 30``, sample proportionally to visit
      counts (τ=1). After that, play the most-visited move (argmax).

    Args:
        network: An :class:`AlphaZeroNetwork` whose weights drive the
            search.
        num_simulations: MCTS simulations per move.
        c_puct: PUCT exploration constant.
        dirichlet_alpha: Dirichlet concentration for root-noise
            injection.
        dirichlet_epsilon: Mixture weight for Dirichlet noise.
        temperature_threshold: Play proportionally until this move
            number, then greedily.
        move_limit: Ply cap. Games past this are treated as draws.
        starting_board: Optional non-default starting position
            (defaults to the standard Shogi starting SFEN).
        device: Inference device.
        rng: Optional NumPy RNG for reproducibility.

    Returns:
        A list of :class:`TrainingExample`, one per position in the
        game (excluding the terminal position). Returns an empty list
        if the ``starting_board`` is already terminal.
    """
    rng = rng if rng is not None else np.random.default_rng()
    board = starting_board if starting_board is not None else shogi.Board()

    evaluate_fn = make_evaluate_fn(network, device=device)

    def is_terminal_here(b: shogi.Board) -> tuple[bool, Optional[float]]:
        return is_terminal_fn(b, move_limit=move_limit)

    mcts = MCTS(
        evaluate_fn=evaluate_fn,
        legal_moves_fn=legal_moves_fn,
        apply_fn=apply_fn,
        is_terminal_fn=is_terminal_here,
        c_puct=c_puct,
    )

    # Record per-position tensors and STMs; z will be backfilled after
    # the game terminates.
    recorded_inputs: list[torch.Tensor] = []
    recorded_policies: list[torch.Tensor] = []
    recorded_masks: list[torch.Tensor] = []
    recorded_stms: list[int] = []

    while True:
        is_term, _ = is_terminal_here(board)
        if is_term:
            break

        root = mcts.search(
            board,
            num_simulations=num_simulations,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            rng=rng,
        )

        position = board_to_position(board)
        stm = position.side_to_move

        # Record the inputs, the τ=1 policy target, and the mask.
        input_tensor = encode_position(position)
        pi_at_tau_one = policy_target(root, temperature=1.0)
        policy_tensor = _policy_dict_to_tensor(pi_at_tau_one, stm)

        our_moves = [shogi_move_to_move(m) for m in board.legal_moves]
        mask_tensor = legal_move_mask(our_moves, stm)

        recorded_inputs.append(input_tensor)
        recorded_policies.append(policy_tensor)
        recorded_masks.append(mask_tensor)
        recorded_stms.append(stm)

        # Sample the actual move to play. Argmax after the temperature
        # threshold for sharper late-game play.
        greedy = board.move_number > temperature_threshold
        chosen = sample_move(root, temperature=1.0, greedy=greedy, rng=rng)
        board = apply_fn(board, chosen)

    # Determine terminal outcome and STM, then backfill z.
    is_term, terminal_value = is_terminal_here(board)
    assert is_term, "loop exited without a terminal board"
    assert terminal_value is not None
    terminal_stm = BLACK if board.turn == shogi.BLACK else WHITE

    zs = _backfill_z(recorded_stms, terminal_stm, terminal_value)

    return [
        TrainingExample(
            input_tensor=inp,
            policy_target=pi,
            value_target=z,
            legal_mask=mask,
        )
        for inp, pi, mask, z in zip(
            recorded_inputs, recorded_policies, recorded_masks, zs
        )
    ]


# ============================================================================
# Replay buffer
# ============================================================================


class ReplayBuffer:
    """FIFO pool of training examples with uniform-random sampling.

    Backed by ``collections.deque(maxlen=capacity)`` so that pushing
    onto a full buffer evicts the oldest example automatically. This
    matches the "sliding window of recent self-play" policy discussed
    in ``AlphaZero_Training_Reference.md`` §2; the capacity should be
    a few generations' worth of self-play to avoid staleness.

    The buffer is *not* thread-safe. For the survey's single-process
    training loop that is fine; if we ever move to distributed
    self-play, we would swap this for a shared-memory buffer with
    locking.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError(f"capacity must be ≥ 1, got {capacity}")
        self._capacity = capacity
        self._buf: deque[TrainingExample] = deque(maxlen=capacity)

    def add(self, example: TrainingExample) -> None:
        """Add one example. Evicts the oldest if the buffer is full."""
        self._buf.append(example)

    def add_many(self, examples: list[TrainingExample]) -> None:
        """Add many examples in order."""
        for e in examples:
            self._buf.append(e)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_full(self) -> bool:
        return len(self._buf) == self._capacity

    def sample(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Return a mini-batch of stacked training tensors.

        Sampling is uniform with replacement. "With replacement"
        matches the AlphaZero paper — at our buffer sizes the chance
        of collision per batch is negligible, and allowing it
        simplifies the logic when ``batch_size > len(self)``.

        Args:
            batch_size: Number of examples in the returned batch.
            rng: Optional NumPy RNG.

        Returns:
            Dict with keys ``"input"`` (B, 45, 9, 9),
            ``"policy"`` (B, 139, 9, 9), ``"value"`` (B,),
            ``"legal_mask"`` (B, 139, 9, 9).

        Raises:
            RuntimeError: if the buffer is empty.
        """
        if not self._buf:
            raise RuntimeError("cannot sample from an empty replay buffer")
        rng = rng if rng is not None else np.random.default_rng()
        n = len(self._buf)
        # rng.choice supports replacement-only when size > n, which
        # matches our intended semantics (always with replacement).
        indices = rng.integers(0, n, size=batch_size)

        inputs = torch.stack([self._buf[i].input_tensor for i in indices])
        policies = torch.stack([self._buf[i].policy_target for i in indices])
        values = torch.tensor(
            [self._buf[i].value_target for i in indices], dtype=torch.float32,
        )
        masks = torch.stack([self._buf[i].legal_mask for i in indices])
        return {
            "input": inputs,
            "policy": policies,
            "value": values,
            "legal_mask": masks,
        }
