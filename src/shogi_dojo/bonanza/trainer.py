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
from torch.utils.data import DataLoader, IterableDataset

from shogi_dojo.alphazero.network import AlphaZeroNetwork, masked_log_softmax

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

        # Step-based checkpointing (default disabled). When set, the
        # trainer also saves a checkpoint every N optimizer steps,
        # regardless of epoch progress. Useful for long-running streaming
        # training where a single epoch can take hours — without it, a
        # crash midway through epoch 1 loses all progress.
        self.save_every_steps = train_cfg.get("save_every_steps")

        # Data-loading parallelism. num_workers > 0 parallelizes the
        # heavy CSA parsing + encoding work across CPU processes so the
        # GPU does not starve while one Python thread does encoding.
        # Default 4 is a safe baseline; on a machine with more cores
        # and a large dataset, 8-16 can noticeably improve GPU
        # utilization.
        self.num_workers = train_cfg.get("num_workers", 4)

        # pin_memory + non_blocking .to(device) gives async host→GPU
        # transfer. Only useful when the device is CUDA; on CPU it is
        # a no-op at best and wasted allocation at worst.
        self.pin_memory = train_cfg.get(
            "pin_memory", self.device.type == "cuda",
        ) and self.device.type == "cuda"

        # Resume state. ``start_epoch`` is the first epoch to run
        # (1-indexed); ``global_step`` is the monotonic optimizer-step
        # counter. Both are mutated by :meth:`resume_from`.
        self.start_epoch = 1
        self.global_step = 0

    def resume_from(self, checkpoint_path: str) -> None:
        """Restore model + optimizer state from a saved checkpoint.

        Sets ``self.start_epoch`` to the epoch *after* the one recorded
        in the checkpoint, and ``self.global_step`` to the saved step
        counter, so the next :meth:`train` call continues from there.

        Step-based checkpoints (``bonanza_step_*.pt``) are saved
        mid-epoch; resuming from one skips the remainder of that epoch
        and restarts at the next one. Exact mid-epoch resume would
        require DataLoader/RNG state that we do not persist.
        """
        ckpt = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True,
        )
        self.network.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = int(ckpt["epoch"]) + 1
        self.global_step = int(ckpt["global_step"])
        logger.info(
            "Resumed from %s (saved epoch=%d, global_step=%d) — "
            "next epoch=%d",
            checkpoint_path, ckpt["epoch"], self.global_step, self.start_epoch,
        )

    def train(self, dataset) -> None:
        """Run the full supervised training loop.

        Args:
            dataset: A map-style :class:`SupervisedDataset` (loads all
                samples into memory), a :class:`StreamingSupervisedDataset`
                (streams lazily from CSA files), or a
                :class:`~shogi_dojo.bonanza.packed.PackedDataset`
                (pre-processed shards, fastest). All are accepted; the
                DataLoader configuration is adjusted automatically.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        is_streaming = isinstance(dataset, IterableDataset)

        # For IterableDataset, shuffle must be False (the dataset itself
        # handles shuffling via its internal buffer).
        # persistent_workers=True keeps workers alive across epochs —
        # saves the reset cost that otherwise runs the streaming
        # dataset's per-worker file shuffle + RNG setup every epoch.
        # Datasets that decode a whole batch at once (PackedDataset)
        # provide their own collate_fn; None means PyTorch's default.
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=not is_streaming,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
        )

        common_info = (
            f"batch_size={self.batch_size}, num_workers={self.num_workers}, "
            f"pin_memory={self.pin_memory}, lambda_value={self.lambda_value:.2f}"
        )
        if is_streaming:
            logger.info(
                "Training: streaming dataset, %d epochs, %s",
                self.epochs, common_info,
            )
        else:
            logger.info(
                "Training: %d samples, %d epochs, %s",
                len(dataset), self.epochs, common_info,
            )

        # Monotonic counter across all epochs — used for step-based
        # checkpointing and logged in every saved checkpoint. When
        # resuming, this picks up where the loaded checkpoint left off.
        global_step = self.global_step

        if self.start_epoch > self.epochs:
            logger.warning(
                "Nothing to do: start_epoch=%d already exceeds configured "
                "epochs=%d. Increase training.epochs in the config to "
                "continue training past the resumed checkpoint.",
                self.start_epoch, self.epochs,
            )
            return

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            total_loss = 0.0
            total_ploss = 0.0
            total_vloss = 0.0
            total_correct = 0
            total_samples = 0
            n_batches = 0

            self.network.train()
            for batch in loader:
                # non_blocking=True lets the GPU transfer overlap with
                # the just-completed step's compute. Only effective
                # when pin_memory was set on the DataLoader.
                inputs = batch["input"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)
                masks = batch["legal_mask"].to(self.device, non_blocking=True)
                value_targets = batch["value_target"].to(self.device, non_blocking=True)

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

                # Step-based checkpoint, fired from inside the inner
                # loop so even mid-epoch crashes are recoverable.
                global_step += 1
                if (
                    self.save_every_steps
                    and global_step % self.save_every_steps == 0
                ):
                    self._save_checkpoint(
                        epoch=epoch,
                        global_step=global_step,
                        label=f"step_{global_step:08d}",
                    )

            elapsed = time.time() - t0
            avg_loss = total_loss / max(n_batches, 1)
            avg_ploss = total_ploss / max(n_batches, 1)
            avg_vloss = total_vloss / max(n_batches, 1)
            accuracy = total_correct / max(total_samples, 1)
            samples_per_sec = total_samples / elapsed if elapsed > 0 else 0.0

            logger.info(
                "Epoch %d/%d  loss=%.4f (policy=%.4f value=%.4f)  "
                "accuracy=%.2f%%  %.0f samples/s  %.1fs",
                epoch, self.epochs,
                avg_loss, avg_ploss, avg_vloss,
                accuracy * 100, samples_per_sec, elapsed,
            )

            if epoch % self.save_every == 0 or epoch == self.epochs:
                self._save_checkpoint(
                    epoch=epoch,
                    global_step=global_step,
                    label=f"epoch_{epoch:04d}",
                )

        logger.info("Training complete. %d epochs, %d steps.", self.epochs, global_step)

    def _save_checkpoint(
        self, *, epoch: int, global_step: int, label: str,
    ) -> None:
        """Save a checkpoint with the given ``label`` in the filename.

        Labels distinguish epoch-based (e.g. ``epoch_0010``) from
        step-based (``step_00005000``) checkpoints. Both payloads
        include the current epoch and global step so a future resume
        implementation can pick either and continue correctly.
        """
        path = os.path.join(self.checkpoint_dir, f"bonanza_{label}.pt")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        logger.info("  checkpoint saved: %s", path)
