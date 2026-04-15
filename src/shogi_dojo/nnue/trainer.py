"""Training loop for Shogi NNUE.

=== How NNUE Training Works ===

The goal is to train the neural network to predict two things:

  1. **Engine score**: How good is this position? (measured in centipawns)
     The network learns to mimic a strong engine's evaluation.

  2. **Game result**: Did this position lead to a win, draw, or loss?
     The network learns long-term positional understanding.

We combine both objectives with a weighted sum:
    total_loss = lambda_score * score_loss + lambda_result * result_loss

=== Loss Functions ===

Both losses use the "sigmoid MSE" approach:

  For score loss:
    loss = MSE(sigmoid(predicted), sigmoid(target_score))

  For result loss:
    loss = MSE(sigmoid(predicted), game_result)

Why sigmoid? The network outputs a raw score (can be any value), but we
want to compare it against probabilities (0 to 1). Sigmoid maps any real
number to the range (0, 1). This also means the network implicitly learns
a win-probability function: sigmoid(score) ≈ probability of winning.

=== Training Schedule ===

We support two learning rate schedulers:
  - Cosine annealing: smoothly decreases LR from initial value to ~0
    over all epochs. Generally preferred for NNUE training.
  - Step decay: multiplies LR by a factor (gamma) every N epochs.

=== Best-Model Tracking ===

During training, we track the lowest validation loss seen so far. Whenever
the validation loss improves (reaches a new minimum), we save the model as
"best_model.pt". This way, even if the model overfits or destabilizes in
later epochs, you always have the best-performing weights available.

If no validation set is provided, best-model tracking uses training loss
instead (less reliable, but still useful as a sanity check).

=== Early Stopping ===

Early stopping prevents wasting time (and risking overfitting) when the
model has stopped improving. It works by counting how many consecutive
epochs have passed without the validation loss reaching a new minimum.
If this count exceeds a configurable "patience" threshold, training stops.

For example, with patience=20: if the best validation loss was at epoch 50
and we reach epoch 71 with no improvement, training stops at epoch 71.

Set patience to 0 to disable early stopping.
"""

from __future__ import annotations

import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import SfenDataset, collate_fn
from .model import NnueNetwork

logger = logging.getLogger(__name__)


def sigmoid_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Score loss: MSE between sigmoid of predicted and sigmoid of target.

    Both predicted and target are raw scores (e.g., in centipawn-scaled units).
    We pass both through sigmoid before computing MSE so that:
      - Large score differences are compressed (diminishing returns)
      - The comparison is in "win probability" space

    Args:
        predicted: Model output scores, shape (batch,).
        target: Target scores (from engine), shape (batch,).

    Returns:
        Scalar loss value.
    """
    return F.mse_loss(torch.sigmoid(predicted), torch.sigmoid(target))


def result_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Game result loss: MSE between sigmoid of predicted and actual game result.

    The predicted score is passed through sigmoid to get a "win probability",
    which is compared against the actual game result (1.0=win, 0.5=draw, 0.0=loss).

    This teaches the network that positions which lead to wins should have
    high scores, and positions that lead to losses should have low scores.

    Args:
        predicted: Model output scores, shape (batch,).
        target: Game results (1.0/0.5/0.0), shape (batch,).

    Returns:
        Scalar loss value.
    """
    return F.mse_loss(torch.sigmoid(predicted), target)


class Trainer:
    """Manages the full training process: data loading, training loop, checkpointing."""

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: Configuration dictionary loaded from YAML. See configs/default.yaml.
        """
        self.cfg = cfg

        # Use GPU if available, otherwise fall back to CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # --- Build the model ---
        model_cfg = cfg['model']
        self.model = NnueNetwork(
            ft_out_dim=model_cfg['ft_out_dim'],
            l1_dim=model_cfg['l1_dim'],
            l2_dim=model_cfg['l2_dim'],
        ).to(self.device)

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {param_count:,}")

        # --- Set up the optimizer ---
        # Adam is the standard choice for NNUE training.
        # weight_decay adds L2 regularization to prevent overfitting.
        train_cfg = cfg['training']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg['lr'],
            weight_decay=train_cfg['weight_decay'],
        )

        # Store training hyperparameters for use in the training loop
        self.epochs = train_cfg['epochs']
        self.lambda_score = train_cfg['lambda_score']    # Weight for score loss
        self.lambda_result = train_cfg['lambda_result']  # Weight for result loss
        self.clip_grad_norm = train_cfg['clip_grad_norm']
        self.checkpoint_dir = train_cfg['checkpoint_dir']
        self.save_every = train_cfg['save_every']

        # Early stopping: stop training if no improvement for `patience` epochs.
        # Set to 0 to disable early stopping (train for all epochs).
        self.early_stopping_patience = train_cfg.get('early_stopping_patience', 0)

        # Best-model tracking state
        self.best_loss = float('inf')  # Lowest loss seen so far
        self.best_epoch = 0            # Which epoch achieved the best loss
        self.epochs_without_improvement = 0  # Counter for early stopping

        self._setup_scheduler(train_cfg)

        # TensorBoard logging (optional — only enabled if log_dir is set).
        # Allows visualizing loss curves, learning rate, and other metrics
        # with `tensorboard --logdir <log_dir>`.
        self.writer = None
        log_dir = train_cfg.get('log_dir')
        if log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard logging to: {log_dir}")
            except ImportError:
                logger.info(
                    "TensorBoard not installed (pip install tensorboard). "
                    "Skipping visualization logging."
                )

    def _setup_scheduler(self, train_cfg: dict):
        """Configure the learning rate scheduler.

        The scheduler adjusts the learning rate during training:
          - "cosine": Smoothly decreases LR following a cosine curve.
                      Starts at the initial LR and approaches 0 at the end.
                      This is the recommended default.
          - "step":   Multiplies LR by gamma every step_size epochs.
                      E.g., with gamma=0.5 and step_size=50, LR is halved
                      at epoch 50, 100, 150, etc.
        """
        if train_cfg['lr_scheduler'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif train_cfg['lr_scheduler'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_cfg['lr_step_size'],
                gamma=train_cfg['lr_gamma'],
            )
        else:
            self.scheduler = None

    def _make_loader(self, path: str) -> DataLoader:
        """Create a DataLoader for a given SFEN data file.

        The DataLoader handles:
          - Batching: groups samples into batches of batch_size
          - Multi-process loading: uses num_workers parallel processes
          - Memory pinning: pre-copies data to GPU-accessible memory (if using CUDA)
          - Custom collation: pads variable-length index tensors (see dataset.py)
        """
        data_cfg = self.cfg['data']
        dataset = SfenDataset(
            path,
            score_scaling=self.cfg['training']['score_scaling'],
        )
        return DataLoader(
            dataset,
            batch_size=data_cfg['batch_size'],
            collate_fn=collate_fn,        # Custom padding for sparse indices
            num_workers=data_cfg['num_workers'],
            pin_memory=(self.device.type == 'cuda'),  # Faster CPU→GPU transfer
        )

    def train(self):
        """Run the full training loop for all epochs.

        Each epoch:
          1. Train on all batches in the training set
          2. (Optional) Evaluate on the validation set
          3. Log metrics (loss, learning rate, elapsed time)
          4. Track best model (save if validation/train loss improves)
          5. Check early stopping condition
          6. Step the learning rate scheduler
          7. Save a periodic checkpoint every save_every epochs
        """
        data_cfg = self.cfg['data']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        train_loader = self._make_loader(data_cfg['train_path'])

        # Validation is optional — only used if a val_path exists
        val_loader = None
        if data_cfg.get('val_path') and os.path.exists(data_cfg['val_path']):
            val_loader = self._make_loader(data_cfg['val_path'])

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            elapsed = time.time() - t0

            msg = f"Epoch {epoch}/{self.epochs}  train_loss={train_loss:.6f}  time={elapsed:.1f}s"

            # Use validation loss for best-model tracking if available,
            # otherwise fall back to training loss.
            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader)
                msg += f"  val_loss={val_loss:.6f}"
                tracked_loss = val_loss
            else:
                tracked_loss = train_loss

            lr = self.optimizer.param_groups[0]['lr']
            msg += f"  lr={lr:.2e}"
            logger.info(msg)

            # --- TensorBoard logging ---
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                if val_loader is not None:
                    self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('LR', lr, epoch)

            # --- Best-model tracking ---
            # If this epoch's loss is the best so far, save it as "best_model.pt".
            # This ensures you always have the best weights even if later epochs
            # overfit or destabilize.
            if tracked_loss < self.best_loss:
                self.best_loss = tracked_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_best_model(epoch, tracked_loss)
            else:
                self.epochs_without_improvement += 1

            # --- Early stopping ---
            # If the loss hasn't improved for `patience` consecutive epochs,
            # there's no point continuing — the model has likely converged or
            # started overfitting. Stop early to save time.
            if (self.early_stopping_patience > 0
                    and self.epochs_without_improvement >= self.early_stopping_patience):
                logger.info(
                    f"Early stopping: no improvement for {self.early_stopping_patience} epochs. "
                    f"Best loss={self.best_loss:.6f} at epoch {self.best_epoch}."
                )
                # Save a final checkpoint before stopping
                self._save_checkpoint(epoch)
                break

            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint periodically and always at the final epoch
            if epoch % self.save_every == 0 or epoch == self.epochs:
                self._save_checkpoint(epoch)

        logger.info(
            f"Training complete. Best loss={self.best_loss:.6f} at epoch {self.best_epoch}. "
            f"Best model saved to: {os.path.join(self.checkpoint_dir, 'best_model.pt')}"
        )

        if self.writer is not None:
            self.writer.close()

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch (one full pass through the training data).

        For each batch:
          1. Forward pass: compute predictions
          2. Compute loss (score loss + result loss)
          3. Backward pass: compute gradients
          4. Clip gradients (prevents exploding gradients)
          5. Optimizer step: update weights

        Returns:
            Average loss over all batches in this epoch.
        """
        self.model.train()  # Enable training mode (affects dropout, batchnorm, etc.)
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            # Move data to the correct device (CPU or GPU)
            stm = batch['stm_indices'].to(self.device)
            nstm = batch['nstm_indices'].to(self.device)
            score = batch['score'].to(self.device)
            game_result = batch['result'].to(self.device)

            # Forward pass: predict evaluation scores
            pred = self.model(stm, nstm).squeeze(-1)  # (batch, 1) → (batch,)

            # Compute combined loss
            loss_s = sigmoid_loss(pred, score)       # How well does it match the engine?
            loss_r = result_loss(pred, game_result)  # How well does it predict game outcome?
            loss = self.lambda_score * loss_s + self.lambda_result * loss_r

            # Backward pass and weight update
            self.optimizer.zero_grad()  # Clear old gradients
            loss.backward()            # Compute new gradients

            # Gradient clipping: if gradients are too large, scale them down.
            # This prevents "exploding gradients" that can destabilize training.
            if self.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()  # Apply gradients to update weights

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _eval_epoch(self, loader: DataLoader) -> float:
        """Run one evaluation epoch (validation set, no gradient updates).

        Same as _train_epoch but:
          - model.eval() disables training-specific behavior
          - torch.no_grad() skips gradient computation (saves memory and time)
          - No optimizer.step() — we're just measuring, not learning

        Returns:
            Average loss over all validation batches.
        """
        self.model.eval()  # Disable training-specific behavior
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():  # Don't compute gradients (faster, less memory)
            for batch in loader:
                stm = batch['stm_indices'].to(self.device)
                nstm = batch['nstm_indices'].to(self.device)
                score = batch['score'].to(self.device)
                game_result = batch['result'].to(self.device)

                pred = self.model(stm, nstm).squeeze(-1)

                loss_s = sigmoid_loss(pred, score)
                loss_r = result_loss(pred, game_result)
                loss = self.lambda_score * loss_s + self.lambda_result * loss_r

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, epoch: int):
        """Save a training checkpoint to disk.

        The checkpoint contains:
          - model_state_dict: all learned weights
          - optimizer_state_dict: optimizer state (momentum buffers, etc.)
          - epoch: which epoch this checkpoint is from
          - best_loss / best_epoch: best-model tracking state (so resume works correctly)

        Checkpoints allow you to:
          - Resume training after interruption (--resume flag)
          - Select the best model for deployment
          - Compare models from different training stages
        """
        path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:04d}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def _save_best_model(self, epoch: int, loss: float):
        """Save the current model as the best model so far.

        This is separate from periodic checkpoints. The file "best_model.pt"
        is always overwritten with the latest best, so you don't accumulate
        multiple best-model files. After training, this file contains the
        single best weights for deployment.
        """
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': loss,
            'best_epoch': epoch,
        }, path)
        logger.info(f"New best model at epoch {epoch} (loss={loss:.6f})")
