"""Tests for shogi_dojo.alphazero.network — ResNet with policy/value heads.

Coverage goals:
  - Architectural building blocks (ResidualBlock) preserve shape and
    behave like a true residual connection (identity is representable).
  - The full AlphaZeroNetwork produces outputs of the right shape,
    dtype, and value range (tanh-bounded value head).
  - Default construction stays within a reasonable parameter budget
    for a "tiny didactic" network.
  - Forward passes work on edge-case batch sizes (including 1).
  - Train/eval mode affects BatchNorm behavior as expected.
  - Backward passes produce non-zero gradients on all learnable
    parameters — verifies that no parameter is accidentally cut off
    from the gradient graph.
  - masked_log_softmax really masks illegal actions (assigns zero
    probability) and renormalizes legal ones.
  - alphazero_loss computes reasonable values on synthetic inputs and
    returns per-term breakdowns for logging.
"""

from __future__ import annotations

import pytest
import torch

from shogi_dojo.alphazero.encoding import (
    NUM_INPUT_CHANNELS,
    NUM_MOVE_PLANES,
    POLICY_DIM,
)
from shogi_dojo.alphazero.network import (
    AlphaZeroNetwork,
    ResidualBlock,
    alphazero_loss,
    masked_log_softmax,
)
from shogi_dojo.nnue.constants import NUM_FILES, NUM_RANKS


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------


class TestResidualBlock:
    def test_preserves_shape(self):
        block = ResidualBlock(num_filters=32)
        block.eval()
        x = torch.randn(2, 32, 9, 9)
        y = block(x)
        assert y.shape == x.shape

    def test_identity_path_is_representable(self):
        # If we zero out all conv weights, the residual connection
        # should still carry the input through (after the final ReLU).
        block = ResidualBlock(num_filters=32)
        with torch.no_grad():
            for p in [block.conv1.weight, block.conv2.weight]:
                p.zero_()
            # Zero the BN learnable params so the BN branch outputs 0
            # exactly (running stats are initialized to allow this).
            for bn in [block.bn1, block.bn2]:
                bn.weight.zero_()
                bn.bias.zero_()
        block.eval()
        x = torch.randn(2, 32, 9, 9).abs()  # non-negative so ReLU preserves
        y = block(x)
        assert torch.allclose(y, x, atol=1e-6)


# ---------------------------------------------------------------------------
# AlphaZeroNetwork — shape and construction
# ---------------------------------------------------------------------------


class TestNetworkConstruction:
    def test_default_construction_succeeds(self):
        net = AlphaZeroNetwork()
        assert net.num_blocks == 4
        assert net.num_filters == 64
        assert net.num_input_channels == NUM_INPUT_CHANNELS
        assert net.num_policy_planes == NUM_MOVE_PLANES

    def test_custom_construction_overrides(self):
        net = AlphaZeroNetwork(
            num_blocks=2, num_filters=32,
            value_head_hidden=16,
        )
        assert net.num_blocks == 2
        assert net.num_filters == 32
        # Tower has exactly num_blocks residual blocks.
        assert len(net.tower) == 2

    def test_default_parameter_count_in_expected_range(self):
        # Sanity check: the ~500K order-of-magnitude estimate for the
        # didactic default should land in a loose range. We assert a
        # wide band to avoid brittleness against small architecture
        # tweaks.
        net = AlphaZeroNetwork()
        n_params = sum(p.numel() for p in net.parameters())
        assert 100_000 < n_params < 1_000_000, (
            f"Expected default parameter count in (100K, 1M); got {n_params:,}"
        )


# ---------------------------------------------------------------------------
# Forward pass — shape, dtype, value range
# ---------------------------------------------------------------------------


class TestNetworkForward:
    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_output_shapes(self, batch_size):
        net = AlphaZeroNetwork(num_blocks=2, num_filters=16)
        net.eval()
        x = torch.randn(batch_size, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
        policy_logits, value = net(x)

        assert policy_logits.shape == (batch_size, NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)
        assert value.shape == (batch_size, 1)

    def test_value_is_in_minus_one_plus_one_range(self):
        # The tanh at the end of the value head guarantees |v| <= 1.
        # (Mathematically the range is open, but float32 tanh can
        # saturate exactly at the boundary for extreme pre-activations.)
        net = AlphaZeroNetwork(num_blocks=2, num_filters=16)
        net.eval()
        x = torch.randn(8, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
        _, value = net(x)
        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)

    def test_forward_pass_does_not_nan(self):
        # Forward on random input with the default architecture must
        # produce finite outputs (no NaN, no inf).
        net = AlphaZeroNetwork()
        net.eval()
        x = torch.randn(4, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
        policy, value = net(x)
        assert torch.isfinite(policy).all()
        assert torch.isfinite(value).all()

    def test_policy_flattens_to_policy_dim(self):
        # After flattening, the policy covers the full action space.
        net = AlphaZeroNetwork(num_blocks=2, num_filters=16)
        net.eval()
        x = torch.randn(3, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
        policy, _ = net(x)
        flat = policy.flatten(start_dim=1)
        assert flat.shape == (3, POLICY_DIM)


# ---------------------------------------------------------------------------
# Train/eval mode behavior
# ---------------------------------------------------------------------------


class TestTrainEvalMode:
    def test_batch_norm_differs_between_modes(self):
        # A network in training mode uses the batch's statistics; in
        # eval mode it uses running statistics. On a single-sample
        # batch, BN with batch stats produces 0 output (variance of a
        # single sample is 0 → divide-by-tiny after epsilon). Running
        # stats are initialized to zero-mean/unit-var, so eval mode
        # gives a non-degenerate output.
        net = AlphaZeroNetwork(num_blocks=1, num_filters=8)

        x = torch.randn(1, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)

        net.train()
        policy_train, _ = net(x)

        net.eval()
        policy_eval, _ = net(x)

        # Outputs should differ between modes (not require exact
        # comparison, just that they're not identical).
        assert not torch.allclose(policy_train, policy_eval)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_all_parameters_receive_gradient(self):
        # Every trainable parameter should have a non-zero gradient
        # after one forward+backward. If some parameter is structurally
        # disconnected (e.g. a head is never used), this test catches it.
        net = AlphaZeroNetwork(num_blocks=2, num_filters=16)
        net.train()

        x = torch.randn(4, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
        policy, value = net(x)

        # A loss that uses both heads so both receive gradient signal.
        loss = policy.pow(2).mean() + value.pow(2).mean()
        loss.backward()

        for name, p in net.named_parameters():
            assert p.grad is not None, f"{name} did not receive a gradient"
            # Some gradient should be non-zero. (Running stats in BN
            # don't count as parameters.)
            assert p.grad.abs().sum().item() > 0, (
                f"{name} has all-zero gradient — possibly disconnected from the loss graph"
            )


# ---------------------------------------------------------------------------
# masked_log_softmax
# ---------------------------------------------------------------------------


class TestMaskedLogSoftmax:
    def test_illegal_positions_get_near_minus_infinity_log_prob(self):
        logits = torch.zeros(1, NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)
        mask = torch.zeros(NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES, dtype=torch.bool)
        # Only one action is legal.
        mask[0, 4, 4] = True

        log_probs = masked_log_softmax(logits, mask)
        # The legal action gets log(1) = 0.
        assert torch.allclose(log_probs[0, 0, 4, 4], torch.tensor(0.0), atol=1e-6)
        # All illegal entries are very negative. Build a mask that
        # excludes only the legal cell and confirm everything else is
        # ≪ 0 (floating-point-y log of ~0 from the -1e9 fill).
        illegal_only = torch.ones_like(log_probs, dtype=torch.bool)
        illegal_only[0, 0, 4, 4] = False
        assert log_probs[illegal_only].max().item() < -100.0

    def test_legal_actions_sum_to_one(self):
        logits = torch.randn(2, NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)
        mask = torch.zeros(NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES, dtype=torch.bool)
        # Three legal actions.
        mask[0, 0, 0] = True
        mask[5, 4, 4] = True
        mask[132, 8, 8] = True

        log_probs = masked_log_softmax(logits, mask)
        # Exponentiate and sum over all (plane, y, x). Illegal entries
        # are ≈ 0 after exp, so the total is dominated by the legal
        # entries and should be ≈ 1 per batch sample.
        probs_total = log_probs.exp().flatten(start_dim=1).sum(dim=1)
        assert torch.allclose(probs_total, torch.ones(2), atol=1e-5)

    def test_accepts_per_batch_mask(self):
        # If the mask already has a batch dimension, it should be used
        # as-is (different legal moves per batch element).
        logits = torch.zeros(2, NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)
        mask = torch.zeros(2, NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES, dtype=torch.bool)
        mask[0, 0, 0, 0] = True
        mask[1, 132, 4, 4] = True

        log_probs = masked_log_softmax(logits, mask)
        assert torch.allclose(log_probs[0, 0, 0, 0], torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(log_probs[1, 132, 4, 4], torch.tensor(0.0), atol=1e-6)


# ---------------------------------------------------------------------------
# alphazero_loss
# ---------------------------------------------------------------------------


class TestAlphaZeroLoss:
    def test_returns_scalar_and_breakdown(self):
        net = AlphaZeroNetwork(num_blocks=1, num_filters=8)
        net.eval()
        x = torch.randn(2, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
        policy, value = net(x)

        # Uniform-random target distribution over a random legal mask.
        mask = torch.zeros(NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES, dtype=torch.bool)
        mask.view(-1)[:10] = True    # first 10 actions legal
        target_policy = torch.zeros(2, NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)
        target_policy.view(2, -1)[:, :10] = 0.1  # uniform over the 10 legal
        target_value = torch.tensor([[1.0], [-1.0]])

        loss, parts = alphazero_loss(policy, value, target_policy, target_value, legal_mask=mask)

        assert loss.dim() == 0   # scalar
        assert set(parts.keys()) == {"value_loss", "policy_loss"}
        assert parts["value_loss"].dim() == 0
        assert parts["policy_loss"].dim() == 0

    def test_zero_target_and_zero_value_gives_zero_loss(self):
        # Sanity: if policy target places 0 mass on any action and the
        # value target matches the prediction, both components should
        # vanish (or be very small). The policy term is 0 by linearity
        # (0 · anything = 0), and the value term is 0 when v == z.
        net = AlphaZeroNetwork(num_blocks=1, num_filters=8)
        net.eval()
        x = torch.randn(1, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
        policy, value = net(x)

        target_policy = torch.zeros(1, NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)
        target_value = value.detach()   # force the value loss to zero

        loss, parts = alphazero_loss(policy, value, target_policy, target_value)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        assert parts["value_loss"].item() == pytest.approx(0.0, abs=1e-6)
        assert parts["policy_loss"].item() == pytest.approx(0.0, abs=1e-6)

    def test_value_weight_scales_value_loss(self):
        net = AlphaZeroNetwork(num_blocks=1, num_filters=8)
        net.eval()
        x = torch.randn(2, NUM_INPUT_CHANNELS, NUM_RANKS, NUM_FILES)
        policy, value = net(x)

        target_policy = torch.zeros(2, NUM_MOVE_PLANES, NUM_RANKS, NUM_FILES)
        target_value = torch.tensor([[1.0], [-1.0]])

        loss_1, parts_1 = alphazero_loss(policy, value, target_policy, target_value, value_weight=1.0)
        loss_2, parts_2 = alphazero_loss(policy, value, target_policy, target_value, value_weight=2.0)

        # With policy target all zeros the policy loss is zero, so the
        # total loss is just value_weight * value_loss. Doubling
        # value_weight should roughly double the total loss.
        assert parts_1["value_loss"].item() == pytest.approx(parts_2["value_loss"].item(), rel=1e-6)
        assert loss_2.item() == pytest.approx(2 * loss_1.item(), rel=1e-5)
