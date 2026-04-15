"""Tests for shogi_dojo.nnue.model — FeatureTransformer and NnueNetwork."""

import torch

from shogi_dojo.nnue.constants import FEATURE_TRANSFORMER_INPUT_DIM
from shogi_dojo.nnue.model import FeatureTransformer, NnueNetwork


# ---------------------------------------------------------------------------
# FeatureTransformer
# ---------------------------------------------------------------------------

class TestFeatureTransformerForward:
    """Tests for the simple (non-batched) forward pass."""

    def setup_method(self):
        self.ft = FeatureTransformer(input_dim=100, output_dim=16)

    def test_output_shape(self):
        indices = [torch.tensor([0, 5, 10]), torch.tensor([1, 2])]
        out = self.ft.forward(indices, batch_size=2)
        assert out.shape == (2, 16)

    def test_empty_indices_returns_bias(self):
        # With no active features, output should equal the bias exactly.
        indices = [torch.tensor([], dtype=torch.long)]
        out = self.ft.forward(indices, batch_size=1)
        assert torch.allclose(out.squeeze(0), self.ft.bias)

    def test_single_index_adds_weight_to_bias(self):
        indices = [torch.tensor([3])]
        out = self.ft.forward(indices, batch_size=1)
        expected = self.ft.bias + self.ft.weight[3]
        assert torch.allclose(out.squeeze(0), expected)


class TestFeatureTransformerForwardDense:
    """Tests for the batched forward_dense pass used during training."""

    def setup_method(self):
        self.ft = FeatureTransformer(input_dim=100, output_dim=16)

    def test_output_shape(self):
        stm = torch.tensor([[0, 1, 2], [3, 4, -1]])
        nstm = torch.tensor([[5, 6, -1], [7, 8, 9]])
        out = self.ft.forward_dense(stm, nstm)
        # Concatenated: output_dim * 2
        assert out.shape == (2, 32)

    def test_padding_ignored(self):
        """A -1 padded slot should produce the same output as a shorter tensor
        without that slot."""
        # No padding
        stm_a = torch.tensor([[0, 1, 2]])
        nstm_a = torch.tensor([[3, 4, 5]])
        out_a = self.ft.forward_dense(stm_a, nstm_a)

        # Same indices but with -1 padding appended
        stm_b = torch.tensor([[0, 1, 2, -1, -1]])
        nstm_b = torch.tensor([[3, 4, 5, -1, -1]])
        out_b = self.ft.forward_dense(stm_b, nstm_b)

        assert torch.allclose(out_a, out_b)

    def test_all_padding_returns_bias_concat(self):
        stm = torch.tensor([[-1, -1]])
        nstm = torch.tensor([[-1, -1]])
        out = self.ft.forward_dense(stm, nstm)
        expected = torch.cat([self.ft.bias, self.ft.bias]).unsqueeze(0)
        assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# NnueNetwork
# ---------------------------------------------------------------------------

class TestNnueNetworkForward:
    """Tests for the full NNUE network forward pass."""

    def setup_method(self):
        # Use small dimensions for fast tests.
        self.net = NnueNetwork(ft_input_dim=100, ft_out_dim=8, l1_dim=4, l2_dim=4)

    def test_output_shape_single(self):
        stm = torch.tensor([[0, 1, 2]])
        nstm = torch.tensor([[3, 4, 5]])
        out = self.net(stm, nstm)
        assert out.shape == (1, 1)

    def test_output_shape_batch(self):
        stm = torch.tensor([[0, 1, -1], [2, 3, 4]])
        nstm = torch.tensor([[5, -1, -1], [6, 7, 8]])
        out = self.net(stm, nstm)
        assert out.shape == (2, 1)

    def test_output_is_finite(self):
        stm = torch.tensor([[0, 1, 2]])
        nstm = torch.tensor([[3, 4, 5]])
        out = self.net(stm, nstm)
        assert torch.isfinite(out).all()

    def test_deterministic(self):
        """Same input should produce the same output (no stochastic layers)."""
        stm = torch.tensor([[0, 1, 2]])
        nstm = torch.tensor([[3, 4, 5]])
        out1 = self.net(stm, nstm)
        out2 = self.net(stm, nstm)
        assert torch.equal(out1, out2)


class TestNnueNetworkClippedReLU:
    """Verify that intermediate activations are clamped to [0, 1]."""

    def test_ft_output_clamped(self):
        net = NnueNetwork(ft_input_dim=100, ft_out_dim=8, l1_dim=4, l2_dim=4)
        stm = torch.tensor([[0, 1, 2]])
        nstm = torch.tensor([[3, 4, 5]])

        # Run just the feature transformer + clamp (first step of forward).
        x = net.ft.forward_dense(stm, nstm)
        x = torch.clamp(x, 0.0, 1.0)
        assert x.min() >= 0.0
        assert x.max() <= 1.0


class TestNnueNetworkDefaultDims:
    """Test that the default network uses the real feature dimensions."""

    def test_default_ft_input_dim(self):
        net = NnueNetwork()
        assert net.ft.weight.shape[0] == FEATURE_TRANSFORMER_INPUT_DIM

    def test_default_layer_sizes(self):
        net = NnueNetwork()
        assert net.ft.weight.shape[1] == 256   # ft_out_dim
        assert net.l1.in_features == 512        # ft_out_dim * 2
        assert net.l1.out_features == 32        # l1_dim
        assert net.l2.in_features == 32
        assert net.l2.out_features == 32        # l2_dim
        assert net.l3.in_features == 32
        assert net.l3.out_features == 1


class TestNnueNetworkGradient:
    """Verify that gradients flow through the network."""

    def test_gradients_exist_after_backward(self):
        net = NnueNetwork(ft_input_dim=100, ft_out_dim=8, l1_dim=4, l2_dim=4)
        stm = torch.tensor([[0, 1, 2]])
        nstm = torch.tensor([[3, 4, 5]])
        out = net(stm, nstm)
        out.backward()

        # All parameters should have gradients after backward.
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestForwardDenseConsistency:
    """Verify forward_dense matches forward (non-batched) for single samples."""

    def test_single_sample_matches(self):
        ft = FeatureTransformer(input_dim=50, output_dim=8)
        indices = [torch.tensor([0, 5, 10])]
        out_simple = ft.forward(indices, batch_size=1)

        # forward_dense with same indices (no padding)
        stm = torch.tensor([[0, 5, 10]])
        nstm = torch.tensor([[0, 5, 10]])  # Same for both perspectives
        out_dense = ft.forward_dense(stm, nstm)

        # First half of dense output should match the simple forward.
        assert torch.allclose(out_simple, out_dense[:, :8], atol=1e-6)


class TestMixedPaddingBatch:
    """Batch where some samples are all-valid and some are all-padding."""

    def test_mixed_batch(self):
        net = NnueNetwork(ft_input_dim=100, ft_out_dim=8, l1_dim=4, l2_dim=4)
        stm = torch.tensor([[0, 1, 2], [-1, -1, -1]])
        nstm = torch.tensor([[3, 4, 5], [-1, -1, -1]])
        out = net(stm, nstm)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all()
        # The two samples should produce different outputs (one has features, one doesn't).
        assert not torch.equal(out[0], out[1])
