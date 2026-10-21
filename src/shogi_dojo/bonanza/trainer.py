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

Optional training features (all off by default, so older configs and
checkpoints behave exactly as before):

- **Mixed precision** (``training.amp``) — run the network in float16
  on CUDA for ~2× throughput. Losses are still computed in float32.
- **Optimizer choice** (``training.optimizer``) — ``adam`` (default),
  ``adamw`` (decoupled weight decay), or ``sgd`` (momentum 0.9, Nesterov).
- **Learning-rate schedule** (``training.lr_schedule``) — linear warmup
  followed by ``constant``, ``cosine`` or ``step`` decay, updated every
  optimizer step.
- **Validation** — pass ``val_dataset`` to :meth:`BonanzaTrainer.train`
  to measure held-out loss/accuracy after every epoch and keep the best
  model in ``bonanza_best.pt``.

See ``doc/bonanza/Bonanza_Reference.md`` for the full discussion.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from shogi_dojo.alphazero.network import AlphaZeroNetwork, masked_log_softmax

logger = logging.getLogger(__name__)

OPTIMIZERS = ("adam", "adamw", "sgd")
LR_SCHEDULES = ("constant", "cosine", "step")


def make_lr_lambda(schedule_cfg: dict, total_steps: Optional[int]) -> Callable[[int], float]:
    """Build the LR multiplier function for ``torch.optim.lr_scheduler.LambdaLR``.

    The returned function maps an optimizer step number (0, 1, 2, ...)
    to a multiplier applied to the base learning rate:

    - **Warmup** (``warmup_steps``, default 0): the multiplier rises
      linearly from ``1 / warmup_steps`` to 1. Large early updates from
      a randomly-initialized network (and from Adam's not-yet-calibrated
      moment estimates) can destabilize training; warmup avoids that.
    - **constant**: stays at 1 after warmup.
    - **cosine**: follows half a cosine wave from 1 down to
      ``min_lr_ratio`` (default 0.01) at ``total_steps``.
    - **step**: multiplies by ``gamma`` (default 0.1) each time training
      passes one of ``milestones`` — fractions of ``total_steps``,
      default ``[0.6, 0.85]``.

    Args:
        schedule_cfg: The ``training.lr_schedule`` config dict.
        total_steps: Total optimizer steps in the run. Required for
            ``cosine`` and ``step``.

    Raises:
        ValueError: Unknown schedule type, or ``total_steps`` missing
            for a schedule that needs it.
    """
    kind = schedule_cfg.get("type", "constant")
    if kind not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule.type must be one of {LR_SCHEDULES}, got {kind!r}")
    warmup = int(schedule_cfg.get("warmup_steps", 0))
    if kind != "constant" and not total_steps:
        raise ValueError(
            f"lr_schedule.type={kind!r} needs the total number of steps. It is "
            "computed from len(dataset) automatically; for streaming datasets "
            "(no len) set lr_schedule.total_steps in the config."
        )

    min_ratio = float(schedule_cfg.get("min_lr_ratio", 0.01))
    gamma = float(schedule_cfg.get("gamma", 0.1))
    milestones = [
        int(f * total_steps) for f in schedule_cfg.get("milestones", [0.6, 0.85])
    ] if total_steps else []

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if kind == "constant":
            return 1.0
        if kind == "cosine":
            span = max(total_steps - warmup, 1)
            progress = min((step - warmup) / span, 1.0)
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        # step decay
        return gamma ** sum(step >= m for m in milestones)

    return lr_lambda


class BonanzaTrainer:
    """Supervised policy + value trainer from professional game records.

    Config dict sections:

    - ``model``: ``num_blocks``, ``num_filters``, ``value_head_hidden``
    - ``training``: ``epochs``, ``batch_size``, ``lr``, ``weight_decay``,
      ``lambda_value``, ``checkpoint_dir``, ``save_every``, and the
      optional ``save_every_steps``, ``num_workers``, ``pin_memory``,
      ``optimizer``, ``amp``, ``lr_schedule``, ``log_every_steps``.
      See ``configs/bonanza/v2.yaml`` for every key with comments.

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
        self.optimizer = self._build_optimizer(train_cfg)

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

        # Progress logging inside an epoch (default disabled). On the
        # full wdoor archive one epoch can take hours; this prints the
        # running loss / accuracy / LR / throughput every N steps.
        self.log_every_steps = train_cfg.get("log_every_steps")

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

        # Mixed precision. float16 autocast is a CUDA feature; on CPU we
        # fall back to float32 so the same config runs everywhere.
        self.amp = bool(train_cfg.get("amp", False)) and self.device.type == "cuda"
        if train_cfg.get("amp") and not self.amp:
            logger.warning("training.amp requested but CUDA is unavailable; using float32.")
        # GradScaler multiplies the loss before backward so that small
        # float16 gradients do not underflow to zero, then un-scales
        # them before the optimizer step. With enabled=False every call
        # is a pass-through.
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        # The LR scheduler needs the number of steps per epoch, which
        # depends on the dataset, so it is built in train(). The config
        # is validated here so mistakes surface before any data loads.
        self.lr_schedule_cfg = train_cfg.get("lr_schedule") or {"type": "constant"}
        make_lr_lambda(self.lr_schedule_cfg, total_steps=1)
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self._pending_scheduler_state: Optional[dict] = None

        # Resume state. ``start_epoch`` is the first epoch to run
        # (1-indexed); ``global_step`` is the monotonic optimizer-step
        # counter; ``best_val_loss`` tracks the best validation loss
        # seen so far. All are restored by :meth:`resume_from`.
        self.start_epoch = 1
        self.global_step = 0
        self.best_val_loss = math.inf

    def _build_optimizer(self, train_cfg: dict) -> torch.optim.Optimizer:
        """Create the optimizer named by ``training.optimizer``.

        - ``adam`` (default, the v1 setting): Adam with L2 weight decay
          folded into the gradient.
        - ``adamw``: Adam with *decoupled* weight decay — the decay is
          applied directly to the weights instead of being mixed into
          Adam's adaptive gradient scaling, which makes it behave like
          real regularization. Usually the better Adam variant.
        - ``sgd``: SGD with Nesterov momentum 0.9 — the classic choice
          for ResNets (and what AlphaZero itself used).
        """
        name = train_cfg.get("optimizer", "adam")
        params = self.network.parameters()
        lr = train_cfg["lr"]
        wd = train_cfg["weight_decay"]
        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
        raise ValueError(f"training.optimizer must be one of {OPTIMIZERS}, got {name!r}")

    def resume_from(self, checkpoint_path: str) -> None:
        """Restore training state from a saved checkpoint.

        Restores the model, optimizer, and — when present in the
        checkpoint — the LR scheduler, AMP grad-scaler and best
        validation loss. Checkpoints from older runs that lack those
        entries still load; the missing pieces start fresh.

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
        if "scaler_state_dict" in ckpt and self.amp:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        # The scheduler object does not exist until train(); keep its
        # state and apply it there.
        self._pending_scheduler_state = ckpt.get("scheduler_state_dict")
        self.best_val_loss = float(ckpt.get("best_val_loss", math.inf))
        self.start_epoch = int(ckpt["epoch"]) + 1
        self.global_step = int(ckpt["global_step"])
        logger.info(
            "Resumed from %s (saved epoch=%d, global_step=%d) — "
            "next epoch=%d",
            checkpoint_path, ckpt["epoch"], self.global_step, self.start_epoch,
        )

    def _make_loader(self, dataset, *, train: bool) -> DataLoader:
        is_streaming = isinstance(dataset, IterableDataset)
        # For IterableDataset, shuffle must be False (the dataset itself
        # handles shuffling via its internal buffer). Validation is never
        # shuffled — order does not affect the averaged metrics.
        # persistent_workers=True keeps workers alive across epochs —
        # saves the reset cost that otherwise runs the streaming
        # dataset's per-worker file shuffle + RNG setup every epoch.
        # Datasets that decode a whole batch at once (PackedDataset)
        # provide their own collate_fn; None means PyTorch's default.
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train and not is_streaming,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
        )

    def _build_scheduler(self, steps_per_epoch: Optional[int]) -> None:
        total_steps = self.lr_schedule_cfg.get("total_steps")
        if total_steps is None and steps_per_epoch is not None:
            total_steps = steps_per_epoch * self.epochs
        lr_lambda = make_lr_lambda(self.lr_schedule_cfg, total_steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        if self._pending_scheduler_state is not None:
            self.scheduler.load_state_dict(self._pending_scheduler_state)
            self._pending_scheduler_state = None
        logger.info(
            "LR schedule: %s (total_steps=%s, warmup_steps=%d), optimizer=%s, amp=%s",
            self.lr_schedule_cfg.get("type", "constant"), total_steps,
            int(self.lr_schedule_cfg.get("warmup_steps", 0)),
            type(self.optimizer).__name__, self.amp,
        )

    def _forward_losses(self, batch: dict) -> tuple[torch.Tensor, ...]:
        """Move a batch to the device and compute (loss, policy, value, n_correct, n)."""
        # non_blocking=True lets the GPU transfer overlap with the
        # just-completed step's compute. Only effective when pin_memory
        # was set on the DataLoader.
        inputs = batch["input"].to(self.device, non_blocking=True)
        targets = batch["target"].to(self.device, non_blocking=True)
        masks = batch["legal_mask"].to(self.device, non_blocking=True)
        value_targets = batch["value_target"].to(self.device, non_blocking=True)

        # Only the network runs under autocast (float16 convolutions are
        # where the speed-up is). The losses below are computed in
        # float32: masked_log_softmax fills illegal logits with -1e9,
        # which does not fit in float16, and summing 11,259 softmax
        # terms is more accurate in float32.
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            policy_logits, value = self.network(inputs)

        # Policy loss: cross-entropy with legal-move masking. Flatten the
        # (B, 139, 9, 9) logits to (B, 11259), apply masked log-softmax,
        # then use NLL loss against the target index.
        log_probs = masked_log_softmax(policy_logits.float(), masks)
        flat_log_probs = log_probs.view(inputs.size(0), -1)
        policy_loss = F.nll_loss(flat_log_probs, targets)

        # Value loss: MSE against game outcome.
        value_loss = F.mse_loss(value.float().squeeze(-1), value_targets)

        loss = policy_loss + self.lambda_value * value_loss
        n_correct = (flat_log_probs.argmax(dim=1) == targets).sum()
        return loss, policy_loss, value_loss, n_correct, targets.size(0)

    @torch.no_grad()
    def evaluate(self, dataset) -> dict:
        """Average loss and top-1 accuracy over ``dataset`` without training.

        Returns:
            Dict with ``loss``, ``policy_loss``, ``value_loss``,
            ``accuracy`` (all sample-weighted means) and ``samples``.
        """
        was_training = self.network.training
        self.network.eval()  # BatchNorm uses running stats, not batch stats
        sums = torch.zeros(4, device=self.device)  # loss, policy, value, correct
        n = 0
        for batch in self._make_loader(dataset, train=False):
            loss, ploss, vloss, correct, b = self._forward_losses(batch)
            sums += torch.stack([loss * b, ploss * b, vloss * b, correct.float()])
            n += b
        self.network.train(was_training)
        loss, ploss, vloss, correct = (sums / max(n, 1)).tolist()
        return {
            "loss": loss, "policy_loss": ploss, "value_loss": vloss,
            "accuracy": correct, "samples": n,
        }

    def train(self, dataset, val_dataset=None) -> None:
        """Run the full supervised training loop.

        Args:
            dataset: A map-style :class:`SupervisedDataset` (loads all
                samples into memory), a :class:`StreamingSupervisedDataset`
                (streams lazily from CSA files), or a
                :class:`~shogi_dojo.bonanza.packed.PackedDataset`
                (pre-processed shards, fastest). All are accepted; the
                DataLoader configuration is adjusted automatically.
            val_dataset: Optional held-out dataset of the same kinds.
                When given, it is evaluated after every epoch; metrics
                are logged and stored in checkpoints, and the model with
                the lowest validation loss is saved to
                ``bonanza_best.pt``. Hold out whole *games* (e.g. via
                ``PackedDataset.split``), not random positions — positions
                from one game are nearly identical, so a position-level
                split would leak training data into validation.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        is_streaming = isinstance(dataset, IterableDataset)
        loader = self._make_loader(dataset, train=True)
        steps_per_epoch = None if is_streaming else len(loader)
        self._build_scheduler(steps_per_epoch)

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
            # Running sums stay on the GPU: calling .item() every step
            # would force the CPU to wait for the GPU each time and stall
            # the pipeline. We only synchronize when logging.
            sums = torch.zeros(4, device=self.device)  # loss, policy, value, correct
            total_samples = 0
            n_batches = 0

            self.network.train()
            for batch in loader:
                loss, policy_loss, value_loss, n_correct, b = self._forward_losses(batch)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                sums += torch.stack([
                    loss.detach(), policy_loss.detach(), value_loss.detach(),
                    n_correct.float(),
                ])
                n_batches += 1
                total_samples += b

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

                if self.log_every_steps and global_step % self.log_every_steps == 0:
                    s = sums.tolist()
                    elapsed = time.time() - t0
                    logger.info(
                        "  step %d  epoch %d [%d/%s]  loss=%.4f (policy=%.4f value=%.4f)  "
                        "accuracy=%.2f%%  lr=%.2e  %.0f samples/s",
                        global_step, epoch, n_batches, steps_per_epoch or "?",
                        s[0] / n_batches, s[1] / n_batches, s[2] / n_batches,
                        100 * s[3] / total_samples, self.optimizer.param_groups[0]["lr"],
                        total_samples / max(elapsed, 1e-9),
                    )

            elapsed = time.time() - t0
            s = sums.tolist()
            avg_loss = s[0] / max(n_batches, 1)
            avg_ploss = s[1] / max(n_batches, 1)
            avg_vloss = s[2] / max(n_batches, 1)
            accuracy = s[3] / max(total_samples, 1)
            samples_per_sec = total_samples / elapsed if elapsed > 0 else 0.0

            logger.info(
                "Epoch %d/%d  loss=%.4f (policy=%.4f value=%.4f)  "
                "accuracy=%.2f%%  lr=%.2e  %.0f samples/s  %.1fs",
                epoch, self.epochs,
                avg_loss, avg_ploss, avg_vloss,
                accuracy * 100, self.optimizer.param_groups[0]["lr"],
                samples_per_sec, elapsed,
            )

            val_metrics = None
            if val_dataset is not None:
                val_metrics = self.evaluate(val_dataset)
                improved = val_metrics["loss"] < self.best_val_loss
                if improved:
                    self.best_val_loss = val_metrics["loss"]
                logger.info(
                    "  validation  loss=%.4f (policy=%.4f value=%.4f)  "
                    "accuracy=%.2f%%  on %d samples%s",
                    val_metrics["loss"], val_metrics["policy_loss"],
                    val_metrics["value_loss"], val_metrics["accuracy"] * 100,
                    val_metrics["samples"], "  (new best)" if improved else "",
                )
                if improved:
                    self._save_checkpoint(
                        epoch=epoch, global_step=global_step, label="best",
                        val_metrics=val_metrics,
                    )

            if epoch % self.save_every == 0 or epoch == self.epochs:
                self._save_checkpoint(
                    epoch=epoch,
                    global_step=global_step,
                    label=f"epoch_{epoch:04d}",
                    val_metrics=val_metrics,
                )

        logger.info("Training complete. %d epochs, %d steps.", self.epochs, global_step)

    def _save_checkpoint(
        self, *, epoch: int, global_step: int, label: str,
        val_metrics: Optional[dict] = None,
    ) -> None:
        """Save a checkpoint with the given ``label`` in the filename.

        Labels distinguish epoch-based (e.g. ``epoch_0010``), step-based
        (``step_00005000``) and best-validation (``best``) checkpoints.
        Every payload includes the current epoch and global step plus
        the optimizer, scheduler and grad-scaler state, so any of them
        can be resumed from.
        """
        path = os.path.join(self.checkpoint_dir, f"bonanza_{label}.pt")
        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.amp:
            payload["scaler_state_dict"] = self.scaler.state_dict()
        if val_metrics is not None:
            payload["val_metrics"] = val_metrics
        torch.save(payload, path)
        logger.info("  checkpoint saved: %s", path)
