#!/usr/bin/env python3
"""Play a match between two AlphaZero-family network checkpoints.

Both checkpoints must use the :class:`AlphaZeroNetwork` architecture —
either AlphaZero-trained (from ``train_alphazero.py``) or Bonanza-trained
(from ``train_bonanza.py``). NNUE checkpoints have a different
architecture and are not supported; use the Rust ``eval_match`` tool
for NNUE-vs-NNUE matches.

Usage:
    # Default 40-game match with 200 MCTS sims per move:
    python scripts/alphazero/match.py \
        --network-a checkpoints/alphazero/az_iter_0010.pt \
        --network-b checkpoints/bonanza/bonanza_epoch_0050.pt

    # Networks of different sizes (e.g. Bonanza v2 10x128 vs v1 4x64).
    # Each network's size is read from its own checkpoint automatically:
    python scripts/alphazero/match.py \
        --network-a checkpoints/bonanza_v2/bonanza_best.pt \
        --network-b checkpoints/bonanza/bonanza_epoch_0050.pt \
        --num-games 200 --device cuda

See doc/alphazero/AlphaZero_Training_Reference.md §6 for guidance on
match size and confidence.
"""

import argparse
import logging

import torch

from shogi_dojo.alphazero.evaluation import run_match
from shogi_dojo.alphazero.network import AlphaZeroNetwork, architecture_from_state_dict


def _load_network(
    path: str,
    num_blocks: int | None = None,
    num_filters: int | None = None,
    value_head_hidden: int | None = None,
) -> AlphaZeroNetwork:
    """Load an AlphaZeroNetwork checkpoint.

    Checkpoints do not record their architecture, but it can be read off
    the weight shapes (see :func:`architecture_from_state_dict`), so by
    default each network gets the size its own checkpoint was trained
    with. Any dimension passed explicitly overrides the detected one (a
    wrong override fails loudly in ``load_state_dict``).
    """
    state_dict = torch.load(path, weights_only=True, map_location="cpu")["model_state_dict"]
    arch = architecture_from_state_dict(state_dict)
    overrides = {
        "num_blocks": num_blocks,
        "num_filters": num_filters,
        "value_head_hidden": value_head_hidden,
    }
    arch.update({k: v for k, v in overrides.items() if v is not None})
    logging.info(
        "  architecture: %d blocks x %d filters, value head %d",
        arch["num_blocks"], arch["num_filters"], arch["value_head_hidden"],
    )
    net = AlphaZeroNetwork(**arch)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero/Bonanza match tool")
    parser.add_argument("--network-a", type=str, required=True,
                        help="Checkpoint path for network A")
    parser.add_argument("--network-b", type=str, required=True,
                        help="Checkpoint path for network B")
    parser.add_argument("--num-games", type=int, default=40,
                        help="Total games to play (rounded up to even). Default 40.")
    parser.add_argument("--num-simulations", type=int, default=200,
                        help="MCTS simulations per move. Default 200.")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--move-limit", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Move sampling temperature. Default 0.1 (near-greedy).")
    parser.add_argument("--num-blocks", type=int, default=None,
                        help="Override residual blocks for both networks. "
                             "Default: read from each checkpoint.")
    parser.add_argument("--num-filters", type=int, default=None,
                        help="Override filters per block for both networks. "
                             "Default: read from each checkpoint.")
    parser.add_argument("--value-head-hidden", type=int, default=None,
                        help="Override value head size for both networks. "
                             "Default: read from each checkpoint.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Inference device (cpu, cuda).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Loading network A from %s", args.network_a)
    net_a = _load_network(
        args.network_a, args.num_blocks, args.num_filters, args.value_head_hidden,
    )
    logging.info("Loading network B from %s", args.network_b)
    net_b = _load_network(
        args.network_b, args.num_blocks, args.num_filters, args.value_head_hidden,
    )

    logging.info(
        "Match config: %d games, %d MCTS sims/move, move_limit=%d, device=%s",
        args.num_games, args.num_simulations, args.move_limit, args.device,
    )

    result = run_match(
        net_a, net_b,
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        move_limit=args.move_limit,
        temperature=args.temperature,
        device=args.device,
    )

    print()
    print("=" * 60)
    print("Match result:")
    print(f"  Network A wins:   {result.wins_a}")
    print(f"  Network B wins:   {result.wins_b}")
    print(f"  Draws:            {result.draws}")
    print(f"  Total games:      {result.total_games}")
    print(f"  A's score share:  {result.win_rate_a * 100:.1f}%")
    print(f"  Elo (A - B):      {result.elo_diff_a_minus_b:+.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
