"""Network-vs-network evaluation matches for AlphaZero-family models.

Plays a configurable number of MCTS-guided games between two networks,
color-swapped for fairness, and reports win/loss/draw counts plus an
Elo-difference estimate. This is the Python analogue of the existing
Rust ``eval_match`` tool for NNUE models, but operates on AlphaZero and
Bonanza networks (both use ``AlphaZeroNetwork``, so the same match
logic covers both paradigms).

Two use cases:

1. **Promotion gating during training**
   (``AlphaZero_Training_Reference.md`` §6): after each training
   iteration, play a short match between the latest network and the
   current best. If the latest wins by a margin (≥55% win rate on
   40 games is the LCZero default), promote it. Otherwise keep the
   previous best and discard.

2. **Research comparisons** (``TODO.md`` Phase 3.6): play long matches
   between networks from different paradigms (Bonanza vs AlphaZero,
   AlphaZero-iter-10 vs AlphaZero-iter-50) to measure Elo progression
   across compute budgets.

Game play is identical to self-play except:

- No Dirichlet noise at the root (we want the clean network judgment).
- Temperature is fixed at a low value (near-greedy play).
- Each "game" is actually two games with colors swapped, to cancel out
  whatever first-move advantage exists.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import shogi
import torch

from shogi_dojo.alphazero.mcts import MCTS, sample_move
from shogi_dojo.alphazero.rules import (
    DEFAULT_MOVE_LIMIT,
    apply_fn,
    is_terminal_fn,
    legal_moves_fn,
    make_evaluate_fn,
)
from shogi_dojo.nnue.constants import BLACK, WHITE

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Outcome of a network-vs-network match.

    Attributes:
        wins_a: Number of games won by network A.
        wins_b: Number of games won by network B.
        draws: Number of drawn games (sennichite, move limit, etc.).
        games_a_as_black: Games played with A as Black (for logging).
        games_b_as_black: Games played with B as Black.
    """

    wins_a: int
    wins_b: int
    draws: int
    games_a_as_black: int = 0
    games_b_as_black: int = 0

    @property
    def total_games(self) -> int:
        return self.wins_a + self.wins_b + self.draws

    @property
    def win_rate_a(self) -> float:
        """A's score share, counting draws as half a win."""
        if self.total_games == 0:
            return 0.5
        return (self.wins_a + 0.5 * self.draws) / self.total_games

    @property
    def elo_diff_a_minus_b(self) -> float:
        """Estimated Elo difference (A minus B) from the score share.

        Uses the standard Elo formula:
            elo_diff = -400 * log10(1 / score - 1)

        Returns +inf if A won everything, -inf if B won everything.
        At 50% score the difference is 0.
        """
        s = self.win_rate_a
        if s <= 0.0:
            return float("-inf")
        if s >= 1.0:
            return float("inf")
        return -400.0 * math.log10(1.0 / s - 1.0)

    def summary(self) -> str:
        """Human-readable one-line summary for logging."""
        return (
            f"A: {self.wins_a}W {self.wins_b}L {self.draws}D  "
            f"(score {self.win_rate_a * 100:.1f}%, "
            f"Elo_A-Elo_B ≈ {self.elo_diff_a_minus_b:+.0f})"
        )


def _play_one_game(
    mcts_black: MCTS,
    mcts_white: MCTS,
    num_simulations: int,
    move_limit: int,
    rng: np.random.Generator,
    temperature: float = 0.1,
) -> Optional[int]:
    """Play one game, return winner (BLACK, WHITE, or None for draw).

    ``mcts_black`` is used when it is Black's turn and ``mcts_white``
    when it is White's turn. Each MCTS has its own network bound via
    its evaluator.
    """
    board = shogi.Board()

    def is_terminal_here(b: shogi.Board) -> tuple[bool, Optional[float]]:
        return is_terminal_fn(b, move_limit=move_limit)

    while True:
        is_term, terminal_value = is_terminal_here(board)
        if is_term:
            # Determine winner. terminal_value is from the STM-at-terminal's
            # perspective: -1 means STM lost (someone else won), 0 means draw.
            if terminal_value == 0.0 or terminal_value is None:
                return None
            terminal_stm = BLACK if board.turn == shogi.BLACK else WHITE
            # STM lost → the other side won.
            return WHITE if terminal_stm == BLACK else BLACK

        mcts = mcts_black if board.turn == shogi.BLACK else mcts_white
        root = mcts.search(
            board,
            num_simulations=num_simulations,
            dirichlet_alpha=None,   # no noise during evaluation
            dirichlet_epsilon=0.0,
            rng=rng,
        )
        # Near-greedy play for sharpness.
        move = sample_move(root, temperature=temperature, greedy=False, rng=rng)
        board = apply_fn(board, move)


def run_match(
    network_a: torch.nn.Module,
    network_b: torch.nn.Module,
    num_games: int = 40,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    move_limit: int = DEFAULT_MOVE_LIMIT,
    temperature: float = 0.1,
    device: torch.device | str = "cpu",
    rng: Optional[np.random.Generator] = None,
) -> MatchResult:
    """Play ``num_games`` games between two networks, color-swapped.

    Games are played in pairs: the first game of each pair has A as
    Black, the second has B as Black. This cancels out whatever
    opening-color advantage exists in Shogi.

    Args:
        network_a: First network.
        network_b: Second network.
        num_games: Total number of games. Rounded up to an even number
            so every game has a color-swapped partner.
        num_simulations: MCTS simulations per move.
        c_puct: PUCT exploration constant.
        move_limit: Ply cap per game.
        temperature: Near-zero for sharp play, 1.0 for diverse play.
            0.1 is a reasonable default for evaluation matches.
        device: Inference device.
        rng: Optional NumPy RNG for reproducibility.

    Returns:
        :class:`MatchResult` with win/loss/draw counts and Elo estimate.
    """
    rng = rng if rng is not None else np.random.default_rng()

    # Build one MCTS per network. They differ only in the evaluate_fn.
    eval_a = make_evaluate_fn(network_a, device=device)
    eval_b = make_evaluate_fn(network_b, device=device)

    mcts_a = MCTS(
        evaluate_fn=eval_a,
        legal_moves_fn=legal_moves_fn,
        apply_fn=apply_fn,
        is_terminal_fn=lambda b: is_terminal_fn(b, move_limit=move_limit),
        c_puct=c_puct,
    )
    mcts_b = MCTS(
        evaluate_fn=eval_b,
        legal_moves_fn=legal_moves_fn,
        apply_fn=apply_fn,
        is_terminal_fn=lambda b: is_terminal_fn(b, move_limit=move_limit),
        c_puct=c_puct,
    )

    # Round num_games up to even so pairs are balanced.
    if num_games % 2 == 1:
        num_games += 1

    wins_a = 0
    wins_b = 0
    draws = 0
    games_a_as_black = 0
    games_b_as_black = 0

    for game_idx in range(num_games):
        a_plays_black = (game_idx % 2 == 0)
        if a_plays_black:
            winner = _play_one_game(
                mcts_black=mcts_a, mcts_white=mcts_b,
                num_simulations=num_simulations,
                move_limit=move_limit,
                rng=rng, temperature=temperature,
            )
            games_a_as_black += 1
        else:
            winner = _play_one_game(
                mcts_black=mcts_b, mcts_white=mcts_a,
                num_simulations=num_simulations,
                move_limit=move_limit,
                rng=rng, temperature=temperature,
            )
            games_b_as_black += 1

        # Attribute the win to the correct network.
        if winner is None:
            draws += 1
        elif winner == BLACK:
            if a_plays_black:
                wins_a += 1
            else:
                wins_b += 1
        else:   # winner == WHITE
            if a_plays_black:
                wins_b += 1
            else:
                wins_a += 1

        if (game_idx + 1) % 10 == 0:
            logger.info(
                "  match progress: %d/%d games complete  "
                "(A:%d W, B:%d W, %d D)",
                game_idx + 1, num_games, wins_a, wins_b, draws,
            )

    return MatchResult(
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
        games_a_as_black=games_a_as_black,
        games_b_as_black=games_b_as_black,
    )
