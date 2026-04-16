"""Tests for shogi_dojo.alphazero.evaluation — network-vs-network matches.

Coverage goals:
  - MatchResult computes win_rate_a correctly for common cases
    (all wins A, all wins B, all draws, 50/50, custom mix).
  - MatchResult.elo_diff_a_minus_b returns the expected Elo values
    for known score shares (50% → 0 Elo, 75% → ~190, 0% → -inf).
  - run_match plays the requested number of games (rounded up to even),
    alternates Black/White assignments, and returns counts consistent
    with the total.
  - The match handles short move_limit games gracefully (ending in
    draws).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from shogi_dojo.alphazero.evaluation import MatchResult, run_match
from shogi_dojo.alphazero.network import AlphaZeroNetwork


# ---------------------------------------------------------------------------
# MatchResult arithmetic
# ---------------------------------------------------------------------------


class TestMatchResult:
    def test_empty_result_gives_half_win_rate(self):
        r = MatchResult(wins_a=0, wins_b=0, draws=0)
        assert r.win_rate_a == 0.5   # documented default for zero games

    def test_all_wins_a_gives_one(self):
        r = MatchResult(wins_a=10, wins_b=0, draws=0)
        assert r.win_rate_a == 1.0

    def test_all_wins_b_gives_zero(self):
        r = MatchResult(wins_a=0, wins_b=10, draws=0)
        assert r.win_rate_a == 0.0

    def test_draws_count_as_half(self):
        r = MatchResult(wins_a=0, wins_b=0, draws=10)
        assert r.win_rate_a == 0.5

    def test_mixed_score(self):
        # 5W 3L 2D → (5 + 1) / 10 = 0.6
        r = MatchResult(wins_a=5, wins_b=3, draws=2)
        assert r.win_rate_a == pytest.approx(0.6)

    def test_total_games(self):
        r = MatchResult(wins_a=3, wins_b=5, draws=2)
        assert r.total_games == 10


class TestEloDifference:
    def test_fifty_percent_is_zero_elo(self):
        r = MatchResult(wins_a=5, wins_b=5, draws=0)
        assert r.elo_diff_a_minus_b == pytest.approx(0.0)

    def test_seventy_five_percent_approx_190_elo(self):
        # Known Elo: 75% score share → about 190.85 Elo.
        r = MatchResult(wins_a=75, wins_b=25, draws=0)
        expected = -400 * math.log10(1.0 / 0.75 - 1.0)
        assert r.elo_diff_a_minus_b == pytest.approx(expected)

    def test_all_wins_is_plus_infinity(self):
        r = MatchResult(wins_a=5, wins_b=0, draws=0)
        assert r.elo_diff_a_minus_b == float("inf")

    def test_all_losses_is_minus_infinity(self):
        r = MatchResult(wins_a=0, wins_b=5, draws=0)
        assert r.elo_diff_a_minus_b == float("-inf")


# ---------------------------------------------------------------------------
# run_match integration
# ---------------------------------------------------------------------------


def _tiny_network() -> AlphaZeroNetwork:
    return AlphaZeroNetwork(num_blocks=1, num_filters=8, value_head_hidden=8)


class TestRunMatchIntegration:
    """Smoke test: two tiny networks play a short match."""

    def test_plays_requested_number_of_games(self):
        # num_games=2, move_limit=4 → games end in draws (no mates possible
        # in 4 plies), match completes fast.
        net_a = _tiny_network()
        net_b = _tiny_network()
        result = run_match(
            net_a, net_b,
            num_games=2,
            num_simulations=2,
            move_limit=4,
            rng=np.random.default_rng(0),
        )
        assert result.total_games == 2

    def test_short_limit_produces_draws(self):
        # At move_limit=4 neither side can checkmate, so all games
        # terminate via the move-limit branch → draws.
        net_a = _tiny_network()
        net_b = _tiny_network()
        result = run_match(
            net_a, net_b,
            num_games=4,
            num_simulations=1,
            move_limit=4,
            rng=np.random.default_rng(0),
        )
        assert result.draws == 4
        assert result.wins_a == 0
        assert result.wins_b == 0

    def test_odd_num_games_rounded_up_to_even(self):
        # Requested 3 → actual 4 (rounded up for balanced color pairing).
        net_a = _tiny_network()
        net_b = _tiny_network()
        result = run_match(
            net_a, net_b,
            num_games=3,
            num_simulations=1,
            move_limit=4,
            rng=np.random.default_rng(0),
        )
        assert result.total_games == 4

    def test_color_assignments_balanced(self):
        # Over an even number of games, each network plays Black exactly
        # half the time.
        net_a = _tiny_network()
        net_b = _tiny_network()
        result = run_match(
            net_a, net_b,
            num_games=4,
            num_simulations=1,
            move_limit=4,
            rng=np.random.default_rng(0),
        )
        assert result.games_a_as_black == 2
        assert result.games_b_as_black == 2
