"""Supervised training loop for Bonanza-method policy learning.

This trainer is the simplest of ShogiDojo's three paradigms: standard
epoch-based supervised learning on a fixed dataset of professional game
records. No MCTS, no self-play, no replay buffer — just a DataLoader
and gradient descent.

The network is the same :class:`AlphaZeroNetwork` used by the AlphaZero
pipeline. Both heads are trained:

- **Policy head** → cross-entropy against the professional's move
- **Value head** → MSE against the game outcome (+1 win / -1 loss / 0 draw)

This produces weights that are architecturally compatible with AlphaZero
and can warm-start self-play training.

See ``doc/bonanza/Bonanza_Reference.md`` for the full discussion.
"""

from __future__ import annotations

import logging
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from shogi_dojo.alphazero.network import AlphaZeroNetwork, masked_log_softmax
from shogi_dojo.bonanza.dataset import SupervisedDataset

logger = logging.getLogger(__name__)


class BonanzaTrainer:
    """Supervised policy + value trainer from professional game records.

    Config dict sections:

    - ``model``: ``num_blocks``, ``num_filters``, ``value_head_hidden``
    - ``training``: ``epochs``, ``batch_size``, ``lr``, ``weight_decay``,
      ``lambda_value``, ``checkpoint_dir``, ``save_every``

    The dataset is passed to :meth:`train` rather than configured in
    the YAML, so the same trainer can be reused across different game
    collections.
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

        param_count = sum(p.numel() for p in self.network.parameters())
        logger.info("Network parameters: %s", f"{param_count:,}")

        train_cfg = cfg["training"]
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )

        self.epochs = train_cfg["epochs"]
        self.batch_size = train_cfg["batch_size"]
        self.lambda_value = train_cfg.get("lambda_value", 1.0)
        self.checkpoint_dir = train_cfg["checkpoint_dir"]
        self.save_every = train_cfg.get("save_every", 10)

    def train(self, dataset: SupervisedDataset) -> None:
        """Run the full supervised training loop.

        Args:
            dataset: A :class:`SupervisedDataset` of ``(position, pro_move)``
                samples, typically built via ``SupervisedDataset.from_csa_files``.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        logger.info(
            "Training: %d samples, %d epochs, batch_size=%d, lambda_value=%.2f",
            len(dataset), self.epochs, self.batch_size, self.lambda_value,
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            total_loss = 0.0
            total_ploss = 0.0
            total_vloss = 0.0
            total_correct = 0
            total_samples = 0
            n_batches = 0

            self.network.train()
            for batch in loader:
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
                masks = batch["legal_mask"].to(self.device)
                value_targets = batch["value_target"].to(self.device)

                policy_logits, value = self.network(inputs)

                # Policy loss: cross-entropy with legal-move masking.
                # Flatten the (B, 139, 9, 9) logits to (B, 11259) and
                # apply masked log-softmax, then use NLL loss against
                # the target index.
                log_probs = masked_log_softmax(policy_logits, masks)
                flat_log_probs = log_probs.view(inputs.size(0), -1)
                policy_loss = F.nll_loss(flat_log_probs, targets)

                # Value loss: MSE against game outcome.
                value_loss = F.mse_loss(value.squeeze(-1), value_targets)

                loss = policy_loss + self.lambda_value * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_ploss += policy_loss.item()
                total_vloss += value_loss.item()
                n_batches += 1

                # Accuracy: top-1 move prediction.
                preds = flat_log_probs.argmax(dim=1)
                total_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)

            elapsed = time.time() - t0
            avg_loss = total_loss / max(n_batches, 1)
            avg_ploss = total_ploss / max(n_batches, 1)
            avg_vloss = total_vloss / max(n_batches, 1)
            accuracy = total_correct / max(total_samples, 1)

            logger.info(
                "Epoch %d/%d  loss=%.4f (policy=%.4f value=%.4f)  "
                "accuracy=%.2f%%  %.1fs",
                epoch, self.epochs,
                avg_loss, avg_ploss, avg_vloss,
                accuracy * 100, elapsed,
            )

            if epoch % self.save_every == 0 or epoch == self.epochs:
                self._save_checkpoint(epoch)

        logger.info("Training complete. %d epochs.", self.epochs)

    def _save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(self.checkpoint_dir, f"bonanza_epoch_{epoch:04d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        logger.info("  checkpoint saved: %s", path)
