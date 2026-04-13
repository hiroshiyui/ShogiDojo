"""HalfKAv2 NNUE neural network model for Shogi.

=== Architecture Overview ===

The network has two main parts:

  1. Feature Transformer (the "big" part, ~5M parameters):
     Converts the sparse input features into a dense vector.
     Think of it as a giant lookup table: for each active feature,
     we look up its learned weight vector and sum them all together.

  2. Evaluation Network (the "small" part, ~2K parameters):
     A compact feedforward network that takes the dense features
     and outputs a single evaluation score.

The full data flow for a single position:

    Position (SFEN)
        ↓
    Feature Extraction (features.py) — produces sparse index lists
        ↓
    ┌─────────────────────────────────────────────┐
    │ Feature Transformer                         │
    │   STM indices  → lookup & sum → 256-dim     │
    │   NSTM indices → lookup & sum → 256-dim     │
    │   Concatenate → 512-dim                     │
    └─────────────────────────────────────────────┘
        ↓ ClippedReLU (clamp to [0, 1])
    ┌─────────────────────────────────────────────┐
    │ Evaluation Network                          │
    │   Linear 512 → 32, ClippedReLU              │
    │   Linear 32  → 32, ClippedReLU              │
    │   Linear 32  → 1  (raw eval score)          │
    └─────────────────────────────────────────────┘
        ↓
    Evaluation Score (positive = good for side-to-move)

=== Why ClippedReLU instead of regular ReLU? ===

ClippedReLU clamps activations to [0, 1] instead of just [0, ∞).
This prevents activations from growing too large, which is important
because NNUE models are often quantized to integers for fast inference
in the game engine. Bounded activations make quantization much easier.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .constants import FEATURE_TRANSFORMER_INPUT_DIM


class FeatureTransformer(nn.Module):
    """Transforms sparse piece features into dense vectors.

    This is conceptually similar to an nn.EmbeddingBag: given a list of
    active feature indices, we look up each index's weight vector and sum
    them all together, then add a bias.

    Mathematically:
        output = bias + sum(weight[i] for i in active_indices)

    Why not use nn.EmbeddingBag directly? Because we need to handle two
    perspectives (STM and NSTM) with shared weights and custom padding logic.

    The weight matrix is large (e.g., 19,638 × 256 ≈ 5M parameters) but
    extremely sparse in practice: only ~40 indices are active per position.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # weight[i] is the learned vector for feature index i.
        # When feature i is "active" (piece present), we add weight[i] to the output.
        self.weight = nn.Parameter(torch.zeros(input_dim, output_dim))
        # bias is always added (represents the "base" evaluation with no pieces).
        self.bias = nn.Parameter(torch.zeros(output_dim))
        # Kaiming initialization for stable training with ReLU-family activations
        nn.init.kaiming_normal_(self.weight, a=0.01, mode='fan_out')

    def forward(self, indices: list[torch.Tensor], batch_size: int) -> torch.Tensor:
        """Forward pass using variable-length sparse feature indices.

        This is the simple, non-batched version. It loops over each sample
        individually. Used mainly for debugging or single-position inference.

        Args:
            indices: A list of 1D tensors, one per sample. Each tensor
                     contains the active feature indices for that sample.
                     Lengths can vary between samples.
            batch_size: Number of samples (= len(indices)).

        Returns:
            Dense output tensor of shape (batch_size, output_dim).
        """
        # Start with bias for every sample
        output = self.bias.unsqueeze(0).expand(batch_size, -1).clone()
        for i in range(batch_size):
            if len(indices[i]) > 0:
                # Look up weight vectors for all active indices, then sum them
                output[i] += self.weight[indices[i]].sum(dim=0)
        return output

    def forward_dense(self, stm_indices: torch.Tensor, nstm_indices: torch.Tensor) -> torch.Tensor:
        """Batched forward pass for both perspectives at once.

        This is the main forward method used during training. It processes
        both the side-to-move and not-side-to-move perspectives in parallel,
        then concatenates their outputs.

        Since different positions have different numbers of active features,
        the index tensors are padded to a uniform length with -1 as a sentinel.
        We mask out these padding positions so they don't affect the output.

        Args:
            stm_indices:  (batch_size, max_active_features) — active feature
                          indices for the side-to-move perspective.
                          Padded with -1 for unused slots.
            nstm_indices: Same format, for the not-side-to-move perspective.

        Returns:
            Concatenated output of shape (batch_size, output_dim * 2).
            First half is STM features, second half is NSTM features.
        """
        batch_size = stm_indices.size(0)

        # Start with bias for every sample (will accumulate piece contributions)
        stm_out = self.bias.unsqueeze(0).expand(batch_size, -1).clone()
        nstm_out = self.bias.unsqueeze(0).expand(batch_size, -1).clone()

        # Create masks: True where index is valid (not -1 padding)
        stm_mask = stm_indices >= 0    # shape: (batch, max_active)
        nstm_mask = nstm_indices >= 0

        # Replace -1 with 0 for safe indexing (we'll zero these out with the mask)
        stm_safe = stm_indices.clamp(min=0)
        nstm_safe = nstm_indices.clamp(min=0)

        # Look up weight vectors for all indices: shape (batch, max_active, output_dim)
        stm_feats = self.weight[stm_safe]
        nstm_feats = self.weight[nstm_safe]

        # Zero out the weight vectors at padding positions (where index was -1)
        # mask shape (batch, max_active) → (batch, max_active, 1) for broadcasting
        stm_feats = stm_feats * stm_mask.unsqueeze(-1).float()
        nstm_feats = nstm_feats * nstm_mask.unsqueeze(-1).float()

        # Sum all active weight vectors for each sample: (batch, output_dim)
        stm_out = stm_out + stm_feats.sum(dim=1)
        nstm_out = nstm_out + nstm_feats.sum(dim=1)

        # Concatenate both perspectives: (batch, output_dim * 2)
        return torch.cat([stm_out, nstm_out], dim=1)


class NnueNetwork(nn.Module):
    """Complete HalfKAv2 NNUE network for Shogi position evaluation.

    This network takes sparse feature indices (from features.py) and outputs
    a single scalar evaluation score. Positive values mean the position is
    good for the side-to-move.

    Architecture:
        Sparse Input → FeatureTransformer → [STM_256 || NSTM_256] (=512)
        → ClippedReLU → Linear(512→32) → ClippedReLU
        → Linear(32→32) → ClippedReLU → Linear(32→1) → Score

    Default configuration: ~5M parameters total
        - Feature Transformer: ~5M (the vast majority)
        - Evaluation layers: ~18K (very small and fast)
    """

    def __init__(
        self,
        ft_input_dim: int = FEATURE_TRANSFORMER_INPUT_DIM,
        ft_out_dim: int = 256,
        l1_dim: int = 32,
        l2_dim: int = 32,
    ):
        """Initialize the NNUE network.

        Args:
            ft_input_dim: Total number of possible sparse features (default: 19,638).
            ft_out_dim: Feature transformer output size per perspective (default: 256).
                        Both perspectives are concatenated → 512 total.
            l1_dim: First hidden layer size (default: 32).
            l2_dim: Second hidden layer size (default: 32).
        """
        super().__init__()
        self.ft = FeatureTransformer(ft_input_dim, ft_out_dim)

        # The evaluation network receives concatenated STM + NSTM features
        concat_dim = ft_out_dim * 2  # 256 * 2 = 512
        self.l1 = nn.Linear(concat_dim, l1_dim)  # 512 → 32
        self.l2 = nn.Linear(l1_dim, l2_dim)       # 32  → 32
        self.l3 = nn.Linear(l2_dim, 1)            # 32  → 1 (scalar score)

        self._init_weights()

    def _init_weights(self):
        """Initialize evaluation layer weights with Kaiming initialization.

        Kaiming init is well-suited for ReLU-family activations (including
        ClippedReLU) because it accounts for the fact that ReLU zeroes out
        negative values, and scales initial weights accordingly to maintain
        stable gradient magnitudes across layers.
        """
        for layer in [self.l1, self.l2, self.l3]:
            nn.init.kaiming_normal_(layer.weight, a=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, stm_indices: torch.Tensor, nstm_indices: torch.Tensor) -> torch.Tensor:
        """Run a forward pass to evaluate one or more positions.

        Args:
            stm_indices:  (batch_size, max_active) — padded feature indices
                          for the side-to-move perspective (padded with -1).
            nstm_indices: (batch_size, max_active) — same for the opponent.

        Returns:
            Evaluation scores of shape (batch_size, 1).
            Positive = good for side-to-move, negative = bad.
        """
        # Step 1: Feature Transformer — sparse indices → dense 512-dim vector
        x = self.ft.forward_dense(stm_indices, nstm_indices)
        x = torch.clamp(x, 0.0, 1.0)  # ClippedReLU: clamp activations to [0, 1]

        # Step 2: Evaluation network — three small linear layers
        x = self.l1(x)
        x = torch.clamp(x, 0.0, 1.0)  # ClippedReLU

        x = self.l2(x)
        x = torch.clamp(x, 0.0, 1.0)  # ClippedReLU

        x = self.l3(x)  # Final output — no activation (raw score)
        return x
