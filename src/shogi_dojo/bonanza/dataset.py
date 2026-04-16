"""Dataset for supervised learning from professional Shogi game records.

This module provides the data pipeline for Bonanza-method training:

    CSA files → parse → replay on shogi.Board → encode → PyTorch Dataset

Each training sample is a ``(position, pro_move, game_outcome)`` triple
extracted from a professional game:

- ``position`` is encoded as a 45-channel tensor via
  :func:`shogi_dojo.alphazero.encoding.encode_position`.
- ``pro_move`` is the flat policy index (in [0, 11259)) of the move the
  professional played, via :func:`encode_move`.
- ``game_outcome`` is +1 / -1 / 0 from the position's STM perspective,
  for the value head (same convention as AlphaZero's ``z``).
- ``legal_mask`` is the boolean (139, 9, 9) tensor for masking illegal
  logits during training.

The parser accepts CSA-format files (the standard machine-readable
format for Shogi game records), parsed by python-shogi's
``shogi.CSA.Parser``.

For the survey's scale (thousands of games, each ~100 positions), all
samples are loaded into memory. A streaming ``IterableDataset`` variant
is straightforward to add later if dataset size grows beyond RAM.
"""

from __future__ import annotations

import logging
from pathlib import Path

import shogi
import shogi.CSA
import torch
from torch.utils.data import Dataset

from shogi_dojo.alphazero.encoding import (
    encode_move,
    encode_position,
    legal_move_mask,
)
from shogi_dojo.alphazero.rules import board_to_position, shogi_move_to_move
from shogi_dojo.nnue.constants import (
    BLACK,
    WHITE,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CSA parsing
# ============================================================================


def parse_csa_string(csa_text: str) -> list[dict]:
    """Parse a CSA-format string into a list of game records.

    Each returned dict has keys:

    - ``"sfen"`` — starting position SFEN (usually the standard start).
    - ``"moves"`` — list of USI move strings (e.g. ``"7g7f"``).
    - ``"winner"`` — ``"b"`` (Black won), ``"w"`` (White won), or
      ``None`` for draws / unknown.

    Args:
        csa_text: The full CSA text, which may contain one or multiple
            games (the CSA format supports multi-game files).

    Returns:
        A list of game dicts. Skips games that fail to parse.
    """
    games = []
    try:
        parsed = shogi.CSA.Parser.parse_str(csa_text)
    except Exception as e:
        logger.warning("CSA parse failed: %s", e)
        return []

    for raw in parsed:
        moves = raw.get("moves", [])
        # Skip degenerate entries (e.g., empty-string input produces a
        # zero-move "game" from the CSA parser).
        if not moves:
            continue
        winner = raw.get("win")
        # Normalize winner: python-shogi returns 'b', 'w', or '-'/None.
        if winner not in ("b", "w"):
            winner = None
        games.append({
            "sfen": raw.get("sfen", shogi.STARTING_SFEN),
            "moves": moves,
            "winner": winner,
        })
    return games


def parse_csa_file(path: str | Path) -> list[dict]:
    """Parse a CSA file into game records. Convenience wrapper."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_csa_string(text)


# ============================================================================
# Game replay → training samples
# ============================================================================


def replay_game(
    game: dict,
    min_ply: int = 1,
    max_ply: int = 512,
) -> list[dict]:
    """Replay a parsed game and extract per-position training samples.

    Args:
        game: A game dict from :func:`parse_csa_string`.
        min_ply: Skip the first ``min_ply - 1`` positions (e.g., skip
            move 1 because it is always the same few openings and does
            not teach much). Default: include from ply 1.
        max_ply: Stop replaying after this many plies. Default: 512.

    Returns:
        A list of sample dicts, each with keys ``"input"``,
        ``"target"``, ``"legal_mask"``, ``"value_target"``.
    """
    board = shogi.Board(game.get("sfen", shogi.STARTING_SFEN))
    moves = game.get("moves", [])
    winner = game.get("winner")

    samples: list[dict] = []

    for ply, usi_str in enumerate(moves, start=1):
        if ply > max_ply:
            break

        # Parse the USI move string.
        try:
            shogi_move = shogi.Move.from_usi(usi_str)
        except Exception:
            logger.debug("Skipping invalid USI move %r at ply %d", usi_str, ply)
            break

        # Verify legality (defensive; professional games should always be legal).
        if shogi_move not in board.legal_moves:
            logger.debug("Illegal move %s at ply %d, stopping replay", usi_str, ply)
            break

        if ply >= min_ply:
            # Encode the position BEFORE the move is applied.
            position = board_to_position(board)
            stm = position.side_to_move

            input_tensor = encode_position(position)

            our_move = shogi_move_to_move(shogi_move)
            target_index = encode_move(our_move, stm)

            legal = [shogi_move_to_move(m) for m in board.legal_moves]
            mask = legal_move_mask(legal, stm)

            # Value target: +1 if this position's STM eventually won,
            # -1 if they lost, 0 for draws / unknown.
            if winner is None:
                value_target = 0.0
            elif (winner == "b" and stm == BLACK) or (winner == "w" and stm == WHITE):
                value_target = 1.0
            else:
                value_target = -1.0

            samples.append({
                "input": input_tensor,
                "target": target_index,
                "legal_mask": mask,
                "value_target": value_target,
            })

        # Apply the move.
        board.push(shogi_move)

    return samples


# ============================================================================
# PyTorch Dataset
# ============================================================================


class SupervisedDataset(Dataset):
    """Map-style dataset of ``(position, pro_move)`` training samples.

    All samples are loaded into memory at construction time. For the
    survey's scale (< 100K samples from a few thousand games) this is
    comfortably within RAM.

    The dataset can be constructed from a list of CSA file paths, a list
    of pre-parsed game dicts, or a list of already-replayed sample dicts.

    Usage:

    .. code-block:: python

        dataset = SupervisedDataset.from_csa_files(["games.csa"])
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        for batch in loader:
            policy_logits, value = network(batch["input"])
            loss = cross_entropy(policy_logits_flat, batch["target"])
    """

    def __init__(self, samples: list[dict]):
        """Construct from pre-built sample dicts.

        Each dict must have keys ``"input"`` (Tensor), ``"target"``
        (int), ``"legal_mask"`` (Tensor), ``"value_target"`` (float).
        """
        self._inputs = torch.stack([s["input"] for s in samples])
        self._targets = torch.tensor([s["target"] for s in samples], dtype=torch.long)
        self._masks = torch.stack([s["legal_mask"] for s in samples])
        self._values = torch.tensor(
            [s["value_target"] for s in samples], dtype=torch.float32,
        )

    @classmethod
    def from_csa_files(
        cls,
        paths: list[str | Path],
        min_ply: int = 1,
        max_ply: int = 512,
    ) -> "SupervisedDataset":
        """Parse CSA files and replay all games into a dataset."""
        all_samples: list[dict] = []
        for path in paths:
            games = parse_csa_file(path)
            for game in games:
                all_samples.extend(replay_game(game, min_ply=min_ply, max_ply=max_ply))
        if not all_samples:
            raise ValueError(
                f"No training samples extracted from {len(paths)} CSA file(s). "
                "Check that the files contain valid CSA games with moves."
            )
        logger.info(
            "Loaded %d samples from %d CSA file(s).",
            len(all_samples), len(paths),
        )
        return cls(all_samples)

    @classmethod
    def from_games(
        cls,
        games: list[dict],
        min_ply: int = 1,
        max_ply: int = 512,
    ) -> "SupervisedDataset":
        """Build from already-parsed game dicts (e.g., from
        :func:`parse_csa_string`)."""
        all_samples: list[dict] = []
        for game in games:
            all_samples.extend(replay_game(game, min_ply=min_ply, max_ply=max_ply))
        if not all_samples:
            raise ValueError("No training samples extracted from the provided games.")
        return cls(all_samples)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input": self._inputs[idx],
            "target": self._targets[idx],
            "legal_mask": self._masks[idx],
            "value_target": self._values[idx],
        }
