"""Tests for scripts/nnue/benchmark.py — NNUE model benchmarking."""

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "nnue"))
from benchmark import benchmark_inference, benchmark_training_step, make_dummy_batch, model_stats

from shogi_dojo.nnue.model import NnueNetwork


class TestModelStats:
    def test_returns_positive_counts(self):
        model = NnueNetwork(ft_input_dim=100, ft_out_dim=8, l1_dim=4, l2_dim=4)
        stats = model_stats(model)
        assert stats["total_params"] > 0
        assert stats["trainable_params"] > 0
        assert stats["size_mb"] > 0

    def test_all_params_trainable(self):
        model = NnueNetwork(ft_input_dim=100, ft_out_dim=8, l1_dim=4, l2_dim=4)
        stats = model_stats(model)
        assert stats["total_params"] == stats["trainable_params"]


class TestMakeDummyBatch:
    def test_shapes(self):
        device = torch.device("cpu")
        stm, nstm = make_dummy_batch(32, 40, device)
        assert stm.shape == (32, 40)
        assert nstm.shape == (32, 40)
        assert stm.device.type == "cpu"


class TestBenchmarkInference:
    """Uses default model dims so indices from make_dummy_batch are in range."""

    def test_returns_results_for_each_batch_size(self):
        model = NnueNetwork()  # Default dims match FEATURE_TRANSFORMER_INPUT_DIM
        device = torch.device("cpu")
        batch_sizes = [1, 16]
        results = benchmark_inference(model, device, batch_sizes, warmup=1, iterations=2)
        assert len(results) == 2
        for r in results:
            assert r["positions_per_sec"] > 0
            assert r["elapsed_s"] > 0

    def test_larger_batch_more_total_positions(self):
        model = NnueNetwork()
        device = torch.device("cpu")
        results = benchmark_inference(model, device, [1, 32], warmup=1, iterations=2)
        assert results[1]["total_positions"] > results[0]["total_positions"]


class TestBenchmarkTrainingStep:
    def test_returns_positive_throughput(self):
        model = NnueNetwork()
        device = torch.device("cpu")
        result = benchmark_training_step(model, device, batch_size=16, warmup=1, iterations=2)
        assert result["samples_per_sec"] > 0
        assert result["steps_per_sec"] > 0
        assert result["total_samples"] == 16 * 2
