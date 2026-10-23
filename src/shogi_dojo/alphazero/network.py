"""AlphaZero-style residual network with policy and value heads for Shogi.

This module defines the two-headed convolutional network that an
AlphaZero-style training loop would fit. The architecture is a faithful
(but *small*) rendition of the design from arXiv:1712.01815:

    Input (B, 45, 9, 9)
        │
        ▼
    Initial conv (3×3, 45 → F filters) → BatchNorm → ReLU
        │
        ▼
    N residual blocks, each:
        conv 3×3 F→F → BN → ReLU
          → conv 3×3 F→F → BN
          → add input skip
          → ReLU
        │
        ▼
    ┌───────────────────────────┬───────────────────────────┐
    ▼                           ▼
    Policy head:                Value head:
      conv 1×1, F → 139 planes    conv 1×1, F → 1 plane
      → logits (B, 139, 9, 9)    → BN → ReLU
                                 → flatten (B, 81)
                                 → Linear 81 → V
                                 → ReLU
                                 → Linear V → 1
                                 → tanh (B, 1)

Defaults — ``num_blocks=4``, ``num_filters=64``, ``value_head_hidden=64`` —
produce a network of roughly 340K parameters. That is *tiny* by
AlphaZero standards (the paper used ~46M parameters). The survey
implementation prioritises readability and CPU-trainability over
strength; sizing up is a configuration change rather than a code
change. See ``doc/alphazero/AlphaZero_Training_Reference.md`` §12 for
the rationale behind the small default.

Input shape and number of input channels are fixed by the encoding
module at import time:

- ``(NUM_INPUT_CHANNELS, 9, 9) == (45, 9, 9)`` per position
- ``NUM_MOVE_PLANES == 139`` policy output planes (so policy logits
  live in ``(B, 139, 9, 9)`` and flatten to ``(B, POLICY_DIM) == (B, 11259)``
  for softmax / cross-entropy).

Legal-move masking is **not** applied inside the network. The forward
pass returns raw logits; callers apply a mask (from
:func:`shogi_dojo.alphazero.encoding.legal_move_mask`) before softmax
so that illegal actions receive zero probability. See
:func:`masked_log_softmax` below for a helper that does this
correctly.

Weight initialization follows the pattern used in the NNUE code
(``shogi_dojo.nnue.model``): Kaiming-normal for conv and linear
weights, zeros for biases, standard PyTorch defaults for BatchNorm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from shogi_dojo.alphazero.encoding import (
    NUM_INPUT_CHANNELS,
    NUM_MOVE_PLANES,
)
from shogi_dojo.nnue.constants import NUM_FILES, NUM_RANKS


# ============================================================================
# Building blocks
# ============================================================================


class ResidualBlock(nn.Module):
    """A single pre-activation residual block.

    Shape is preserved: ``(B, F, 9, 9)`` in, ``(B, F, 9, 9)`` out. The
    block is composed of:

        y = ReLU(BN(conv3x3(x)))
        y = BN(conv3x3(y))
        return ReLU(y + x)

    Conv kernels are 3×3 with padding 1 so the spatial dimensions are
    exactly preserved. This is the classical "post-activation"
    formulation from He et al. (2015); variants like pre-activation
    ResNet (He et al. 2016) are equally valid but we stick with the
    original AlphaZero choice for didactic parity with the paper.
    """

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + residual)


# ============================================================================
# The full network
# ============================================================================


class AlphaZeroNetwork(nn.Module):
    """Two-headed residual policy/value network for Shogi.

    Default size is intentionally small (~340K parameters) so the
    network can be trained on a single CPU or consumer GPU for
    research-survey purposes. See the module docstring for the
    architecture diagram and parameter-count rationale.

    Args:
        num_blocks: Number of residual blocks in the tower. Paper used
            19 or 39; we default to 4.
        num_filters: Number of convolution filters per layer (the
            "width" of the residual tower). Paper used 256; we default
            to 64.
        value_head_hidden: Size of the value head's single hidden
            linear layer. Paper used 256; we default to 64.
        num_input_channels: Number of input planes. Defaults to
            ``NUM_INPUT_CHANNELS`` from :mod:`shogi_dojo.alphazero.encoding`
            (currently 45 in the simplified per-position encoding). If
            you extend the encoder to include history planes, pass the
            new channel count here.
        num_policy_planes: Number of output planes in the policy head.
            Defaults to ``NUM_MOVE_PLANES`` (139), matching the paper's
            Shogi action space.
    """

    def __init__(
        self,
        num_blocks: int = 4,
        num_filters: int = 64,
        value_head_hidden: int = 64,
        num_input_channels: int = NUM_INPUT_CHANNELS,
        num_policy_planes: int = NUM_MOVE_PLANES,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.num_input_channels = num_input_channels
        self.num_policy_planes = num_policy_planes

        # --- Stem: initial 3×3 conv that projects the input planes up
        # to the trunk width.
        self.stem_conv = nn.Conv2d(
            num_input_channels, num_filters,
            kernel_size=3, padding=1, bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(num_filters)

        # --- Residual tower ---
        self.tower = nn.ModuleList(
            ResidualBlock(num_filters) for _ in range(num_blocks)
        )

        # --- Policy head ---
        # A single 1×1 conv projects the trunk output directly into the
        # action plane count. The output is (B, num_policy_planes, 9, 9);
        # we flatten to (B, POLICY_DIM) when computing cross-entropy.
        self.policy_conv = nn.Conv2d(
            num_filters, num_policy_planes, kernel_size=1, bias=True,
        )

        # --- Value head ---
        # 1×1 conv down to a single plane, BN+ReLU, flatten to 81, then
        # two FC layers ending in a tanh for a scalar in [-1, +1].
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(NUM_RANKS * NUM_FILES, value_head_hidden)
        self.value_fc2 = nn.Linear(value_head_hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-normal for conv/linear weights, zeros for biases.

        Mirrors the approach in ``shogi_dojo.nnue.model`` so that both
        paradigms start from a consistent initialization style. For
        ReLU-family activations Kaiming-normal keeps pre-activation
        variance roughly stable across layers, which is important for
        the deep residual tower.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # Standard PyTorch defaults for BatchNorm are fine:
                # weight=1, bias=0, running stats tracked.
                pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the two-headed forward pass.

        Args:
            x: Input tensor of shape ``(B, num_input_channels, 9, 9)``,
                typically produced by ``encode_position`` and stacked.

        Returns:
            A tuple ``(policy_logits, value)`` where:

            - ``policy_logits`` has shape ``(B, num_policy_planes, 9, 9)``
              and contains raw (unnormalized, unmasked) logits. To get a
              probability distribution over legal moves, use
              :func:`masked_log_softmax` with a mask from
              :func:`shogi_dojo.alphazero.encoding.legal_move_mask`.

            - ``value`` has shape ``(B, 1)`` and lies in ``[-1, +1]``
              due to the final tanh. Interpretable as the network's
              prediction of the terminal game outcome from the
              side-to-move's perspective (+1 = certain win, −1 =
              certain loss).
        """
        # --- Stem ---
        y = F.relu(self.stem_bn(self.stem_conv(x)))

        # --- Residual tower ---
        for block in self.tower:
            y = block(y)

        # --- Policy head ---
        policy_logits = self.policy_conv(y)  # (B, 139, 9, 9)

        # --- Value head ---
        v = F.relu(self.value_bn(self.value_conv(y)))   # (B, 1, 9, 9)
        v = v.flatten(start_dim=1)                       # (B, 81)
        v = F.relu(self.value_fc1(v))                    # (B, hidden)
        v = torch.tanh(self.value_fc2(v))                # (B, 1)

        return policy_logits, v


def architecture_from_state_dict(state_dict: dict) -> dict[str, int]:
    """Read ``num_blocks`` / ``num_filters`` / ``value_head_hidden`` from weights.

    Checkpoints store only weights, not the constructor arguments, but the
    weight shapes pin the architecture down completely:

    - ``num_filters``: output channels of the stem conv,
      ``stem_conv.weight`` has shape ``(filters, 45, 3, 3)``.
    - ``num_blocks``: how many ``tower.<i>.`` residual blocks exist.
    - ``value_head_hidden``: rows of ``value_fc1.weight``, shape
      ``(hidden, 81)``.

    This lets tools load any AlphaZero-architecture checkpoint (e.g. the
    4×64 v1 and 10×128 v2 Bonanza nets) without being told its size::

        net = AlphaZeroNetwork(**architecture_from_state_dict(sd))
        net.load_state_dict(sd)

    Raises:
        KeyError: if the state dict is not from an ``AlphaZeroNetwork``.
    """
    for key in ("stem_conv.weight", "value_fc1.weight"):
        if key not in state_dict:
            raise KeyError(f"{key!r} missing: not an AlphaZeroNetwork state dict")
    block_ids = {
        int(key.split(".")[1]) for key in state_dict if key.startswith("tower.")
    }
    return {
        "num_blocks": len(block_ids),
        "num_filters": int(state_dict["stem_conv.weight"].shape[0]),
        "value_head_hidden": int(state_dict["value_fc1.weight"].shape[0]),
    }


# ============================================================================
# Loss computation helpers
# ============================================================================


def masked_log_softmax(
    policy_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    neg_inf: float = -1e9,
) -> torch.Tensor:
    """Log-softmax over the policy, with illegal actions zeroed out.

    Illegal logits are set to a large negative number before softmax,
    so they contribute ~0 probability mass and 0 gradient. This is the
    standard way to combine a policy head's output with a legal-move
    mask without requiring a custom softmax implementation.

    Args:
        policy_logits: Raw policy head output, shape
            ``(B, num_planes, 9, 9)`` or any shape that matches
            ``legal_mask`` on dimensions 1..n.
        legal_mask: Boolean tensor, same trailing shape as
            ``policy_logits``, broadcastable over the batch dimension.
            ``True`` = legal, ``False`` = illegal.
        neg_inf: The negative value to substitute for illegal logits.
            Must be very negative (so softmax gives ~0) but not so
            negative that softmax underflows to NaN. The default
            ``-1e9`` is the standard safe choice.

    Returns:
        Log-softmax values, same shape as ``policy_logits``. Illegal
        entries will be very negative (≈ log(0) = −∞ in floats). Legal
        entries sum to 1 when exponentiated.
    """
    # Broadcast the mask across the batch dimension if needed.
    if legal_mask.dim() == policy_logits.dim() - 1:
        legal_mask = legal_mask.unsqueeze(0).expand_as(policy_logits)
    masked_logits = policy_logits.masked_fill(~legal_mask, neg_inf)

    # Flatten all non-batch dims for a single log-softmax call, then
    # reshape back. This avoids shape-specific code for the 139×9×9
    # layout.
    b = masked_logits.size(0)
    flat = masked_logits.view(b, -1)
    log_probs = F.log_softmax(flat, dim=1)
    return log_probs.view_as(policy_logits)


def alphazero_loss(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    legal_mask: torch.Tensor | None = None,
    value_weight: float = 1.0,
    policy_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the combined AlphaZero policy + value loss.

    ``L = value_weight · MSE(v, z) − policy_weight · Σ π · log p``

    L2 weight regularization is *not* included here — PyTorch's
    optimizer-level ``weight_decay`` is the usual place to apply it,
    matching the NNUE training code.

    Args:
        policy_logits: Raw policy head output, ``(B, 139, 9, 9)``.
        value: Value head output, ``(B, 1)`` or ``(B,)``.
        target_policy: MCTS visit distribution, ``(B, 139, 9, 9)``.
            Must be a valid probability distribution per sample (sums
            to 1 along the non-batch dims).
        target_value: Terminal game outcome ``z`` in ``[-1, +1]``,
            shape ``(B, 1)`` or ``(B,)``.
        legal_mask: Optional boolean mask, ``(B, 139, 9, 9)`` or
            ``(139, 9, 9)``. If provided, illegal logits are masked out
            before log-softmax. Omit to skip masking (useful for unit
            tests of the loss itself, not recommended in real training).
        value_weight: Multiplier on the value-MSE term. Default 1.0
            (paper default); some implementations use 1.5 to boost the
            value head's signal.
        policy_weight: Multiplier on the policy cross-entropy term.
            Default 1.0.

    Returns:
        A tuple ``(total_loss, parts)`` where ``parts`` is a dict with
        the individual components (``'value_loss'``, ``'policy_loss'``)
        for logging.
    """
    # --- Policy loss: cross-entropy between MCTS π and network p_θ. ---
    if legal_mask is not None:
        log_probs = masked_log_softmax(policy_logits, legal_mask)
    else:
        b = policy_logits.size(0)
        log_probs = F.log_softmax(policy_logits.view(b, -1), dim=1).view_as(policy_logits)

    # Per-sample cross-entropy, then mean over the batch.
    # (Negative so that minimizing the loss pulls p_θ toward π.)
    policy_loss = -(target_policy * log_probs).view(policy_logits.size(0), -1).sum(dim=1).mean()

    # --- Value loss: MSE between the network's v and the terminal z. ---
    v = value.view(-1)
    z = target_value.view(-1)
    value_loss = F.mse_loss(v, z)

    total = value_weight * value_loss + policy_weight * policy_loss
    return total, {"value_loss": value_loss.detach(), "policy_loss": policy_loss.detach()}
