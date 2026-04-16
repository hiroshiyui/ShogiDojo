"""AlphaZero training loop: self-play → replay buffer → gradient descent.

This module mirrors :mod:`shogi_dojo.nnue.trainer` in structure (a
``Trainer`` class that owns the network, optimizer, and training
state), but the inner loop is fundamentally different:

- **NNUE** reads from a fixed dataset on disk and does epoch-based
  training with validation-loss tracking.
- **AlphaZero** *generates* its own data via self-play, stores it in
  a replay buffer, and trains indefinitely in *iterations* (each
  iteration = self-play games + training steps).

The training loop follows the pseudocode in
``doc/alphazero/AlphaZero_Training_Reference.md`` §5:

    for iteration in range(num_iterations):
        # 1. Self-play: produce fresh training examples
        for _ in range(games_per_iteration):
            examples = play_game(network, ...)
            replay_buffer.add_many(examples)

        # 2. Train: sample mini-batches from the buffer
        for _ in range(training_steps_per_iteration):
            batch = replay_buffer.sample(batch_size)
            loss = alphazero_loss(network(batch), ...)
            optimizer.step(loss)

        # 3. Checkpoint
        save_checkpoint(iteration)

Evaluation-match promotion gating (the
``AlphaZero_Training_Reference.md`` §6 "only promote if win rate ≥ 55%"
policy) is **not implemented** in this first version. Every iteration
saves a checkpoint, and the latest weights are used for the next
iteration's self-play. This keeps Phase 2.7c focused on the wiring;
promotion gating can be layered on later without changing the
architecture.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np
import torch

from shogi_dojo.alphazero.network import AlphaZeroNetwork, alphazero_loss
from shogi_dojo.alphazero.self_play import ReplayBuffer, play_game

logger = logging.getLogger(__name__)


class AlphaZeroTrainer:
    """Drives the self-play + training loop from a YAML config.

    The config dict has four top-level sections:

    - ``model``: ``num_blocks``, ``num_filters``, ``value_head_hidden``
    - ``self_play``: ``games_per_iteration``, ``num_simulations``,
      ``c_puct``, ``dirichlet_alpha``, ``dirichlet_epsilon``,
      ``temperature_threshold``, ``move_limit``
    - ``training``: ``iterations``, ``steps_per_iteration``,
      ``batch_size``, ``lr``, ``weight_decay``, ``checkpoint_dir``,
      ``save_every``
    - ``buffer``: ``capacity``

    See ``configs/alphazero/tiny.yaml`` for a fully commented reference.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        model_cfg = cfg["model"]
        self.network = AlphaZeroNetwork(
            num_blocks=model_cfg["num_blocks"],
            num_filters=model_cfg["num_filters"],
            value_head_hidden=model_cfg.get("value_head_hidden", 64),
        ).to(self.device)

        # Optional warm-start: load weights from a prior checkpoint
        # (e.g., a Bonanza-trained supervised network). Because Bonanza
        # uses the same AlphaZeroNetwork architecture, its state_dict
        # is directly compatible — no conversion needed.
        warm_start_path = model_cfg.get("warm_start")
        if warm_start_path:
            logger.info("Warm-starting from checkpoint: %s", warm_start_path)
            ckpt = torch.load(warm_start_path, weights_only=True, map_location=self.device)
            self.network.load_state_dict(ckpt["model_state_dict"])

        param_count = sum(p.numel() for p in self.network.parameters())
        logger.info("Network parameters: %s", f"{param_count:,}")

        train_cfg = cfg["training"]
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )

        self.iterations = train_cfg["iterations"]
        self.steps_per_iteration = train_cfg["steps_per_iteration"]
        self.batch_size = train_cfg["batch_size"]
        self.checkpoint_dir = train_cfg["checkpoint_dir"]
        self.save_every = train_cfg.get("save_every", 1)

        buf_cfg = cfg["buffer"]
        self.replay_buffer = ReplayBuffer(capacity=buf_cfg["capacity"])

        self.sp_cfg = cfg["self_play"]

    # -----------------------------------------------------------------

    def train(self, rng: Optional[np.random.Generator] = None) -> None:
        """Run the full training loop for ``self.iterations`` iterations.

        Each iteration:
          1. Generate self-play games and push into the replay buffer.
          2. Sample mini-batches and apply gradient steps.
          3. Log metrics.
          4. Save a checkpoint every ``save_every`` iterations (and
             always on the last one).
        """
        rng = rng if rng is not None else np.random.default_rng()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        for iteration in range(1, self.iterations + 1):
            t0 = time.time()

            # --- Self-play ---
            games = self.sp_cfg["games_per_iteration"]
            total_examples = 0
            for game_idx in range(games):
                examples = play_game(
                    network=self.network,
                    num_simulations=self.sp_cfg["num_simulations"],
                    c_puct=self.sp_cfg["c_puct"],
                    dirichlet_alpha=self.sp_cfg["dirichlet_alpha"],
                    dirichlet_epsilon=self.sp_cfg["dirichlet_epsilon"],
                    temperature_threshold=self.sp_cfg["temperature_threshold"],
                    move_limit=self.sp_cfg["move_limit"],
                    device=self.device,
                    rng=rng,
                )
                self.replay_buffer.add_many(examples)
                total_examples += len(examples)

            selfplay_time = time.time() - t0
            logger.info(
                "Iteration %d/%d  self-play: %d games, %d examples (%.1fs)",
                iteration, self.iterations, games, total_examples, selfplay_time,
            )

            # --- Training steps ---
            if len(self.replay_buffer) == 0:
                logger.warning("Replay buffer is empty after self-play; skipping training steps.")
                continue

            t1 = time.time()
            total_loss = 0.0
            total_vloss = 0.0
            total_ploss = 0.0

            self.network.train()
            use_non_blocking = self.device.type == "cuda"
            for _ in range(self.steps_per_iteration):
                batch = self.replay_buffer.sample(self.batch_size, rng=rng)
                # non_blocking transfers overlap with compute on CUDA
                # devices. On CPU this flag is a no-op.
                inputs = batch["input"].to(self.device, non_blocking=use_non_blocking)
                target_policy = batch["policy"].to(self.device, non_blocking=use_non_blocking)
                target_value = batch["value"].to(self.device, non_blocking=use_non_blocking)
                mask = batch["legal_mask"].to(self.device, non_blocking=use_non_blocking)

                policy_logits, value = self.network(inputs)
                loss, parts = alphazero_loss(
                    policy_logits, value,
                    target_policy, target_value,
                    legal_mask=mask,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_vloss += parts["value_loss"].item()
                total_ploss += parts["policy_loss"].item()

            n = self.steps_per_iteration
            train_time = time.time() - t1
            logger.info(
                "  train: %d steps  loss=%.4f (value=%.4f policy=%.4f)  %.1fs",
                n, total_loss / n, total_vloss / n, total_ploss / n, train_time,
            )

            # --- Checkpoint ---
            if iteration % self.save_every == 0 or iteration == self.iterations:
                self._save_checkpoint(iteration)

        logger.info("Training complete. %d iterations.", self.iterations)

    def _save_checkpoint(self, iteration: int) -> None:
        path = os.path.join(self.checkpoint_dir, f"az_iter_{iteration:04d}.pt")
        torch.save({
            "iteration": iteration,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "buffer_size": len(self.replay_buffer),
        }, path)
        logger.info("  checkpoint saved: %s", path)
