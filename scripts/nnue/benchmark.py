#!/usr/bin/env python3
"""Benchmark NNUE model inference and training throughput.

Measures:
  - Inference speed (positions/second) for different batch sizes
  - Training throughput (samples/second) for a single epoch
  - Model memory footprint (parameters and estimated size)

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --config configs/small.yaml
    python scripts/benchmark.py --device cpu
"""

from __future__ import annotations

import argparse
import time

import torch
import yaml

from shogi_dojo.nnue.constants import FEATURE_TRANSFORMER_INPUT_DIM
from shogi_dojo.nnue.model import NnueNetwork
from shogi_dojo.nnue.trainer import sigmoid_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark NNUE model performance.")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Config YAML to read model dimensions from (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to benchmark on (default: auto-detect cuda/cpu).",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup iterations before timing (default: 10).",
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of timed iterations (default: 100).",
    )
    return parser.parse_args()


def make_dummy_batch(
    batch_size: int, num_features: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a dummy batch of random sparse feature indices."""
    stm = torch.randint(0, FEATURE_TRANSFORMER_INPUT_DIM, (batch_size, num_features), device=device)
    nstm = torch.randint(0, FEATURE_TRANSFORMER_INPUT_DIM, (batch_size, num_features), device=device)
    return stm, nstm


def benchmark_inference(
    model: NnueNetwork,
    device: torch.device,
    batch_sizes: list[int],
    warmup: int,
    iterations: int,
) -> list[dict]:
    """Benchmark inference speed for various batch sizes."""
    results = []
    model.eval()

    for bs in batch_sizes:
        stm, nstm = make_dummy_batch(bs, 40, device)

        # Warmup.
        with torch.no_grad():
            for _ in range(warmup):
                model(stm, nstm)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed run.
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                model(stm, nstm)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_positions = bs * iterations
        pos_per_sec = total_positions / elapsed
        results.append({
            "batch_size": bs,
            "total_positions": total_positions,
            "elapsed_s": elapsed,
            "positions_per_sec": pos_per_sec,
        })

    return results


def benchmark_training_step(
    model: NnueNetwork,
    device: torch.device,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> dict:
    """Benchmark a single training step (forward + backward + optimizer step)."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    stm, nstm = make_dummy_batch(batch_size, 40, device)
    target_score = torch.randn(batch_size, device=device)

    # Warmup.
    for _ in range(warmup):
        pred = model(stm, nstm).squeeze(-1)
        loss = sigmoid_loss(pred, target_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed run.
    t0 = time.perf_counter()
    for _ in range(iterations):
        pred = model(stm, nstm).squeeze(-1)
        loss = sigmoid_loss(pred, target_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_samples = batch_size * iterations
    return {
        "batch_size": batch_size,
        "total_samples": total_samples,
        "elapsed_s": elapsed,
        "samples_per_sec": total_samples / elapsed,
        "steps_per_sec": iterations / elapsed,
    }


def model_stats(model: NnueNetwork) -> dict:
    """Compute model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": size_mb,
    }


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = NnueNetwork(
        ft_out_dim=model_cfg["ft_out_dim"],
        l1_dim=model_cfg["l1_dim"],
        l2_dim=model_cfg["l2_dim"],
    ).to(device)

    stats = model_stats(model)

    print("=== Model Info ===")
    print(f"  Config:      {args.config}")
    print(f"  Device:      {device}")
    print(f"  Parameters:  {stats['total_params']:,}")
    print(f"  Size:        {stats['size_mb']:.1f} MB")
    print(f"  FT:          {FEATURE_TRANSFORMER_INPUT_DIM} -> {model_cfg['ft_out_dim']}")
    print(f"  L1:          {model_cfg['ft_out_dim'] * 2} -> {model_cfg['l1_dim']}")
    print(f"  L2:          {model_cfg['l1_dim']} -> {model_cfg['l2_dim']}")
    print(f"  L3:          {model_cfg['l2_dim']} -> 1")
    print()

    # Inference benchmark.
    batch_sizes = [1, 64, 256, 1024, 4096, 8192]
    print(f"=== Inference Benchmark (warmup={args.warmup}, iterations={args.iterations}) ===")
    print(f"  {'Batch':>8}  {'Pos/sec':>12}  {'Time (s)':>10}")
    print(f"  {'-----':>8}  {'-------':>12}  {'--------':>10}")

    inf_results = benchmark_inference(model, device, batch_sizes, args.warmup, args.iterations)
    for r in inf_results:
        print(f"  {r['batch_size']:>8}  {r['positions_per_sec']:>12,.0f}  {r['elapsed_s']:>10.3f}")
    print()

    # Training step benchmark.
    train_bs = cfg["data"].get("batch_size", 8192)
    print(f"=== Training Step Benchmark (batch_size={train_bs}) ===")
    train_result = benchmark_training_step(model, device, train_bs, args.warmup, args.iterations)
    print(f"  Samples/sec:  {train_result['samples_per_sec']:,.0f}")
    print(f"  Steps/sec:    {train_result['steps_per_sec']:.1f}")
    print(f"  Time:         {train_result['elapsed_s']:.3f}s ({args.iterations} steps)")
    print()

    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print("=== GPU Memory ===")
        print(f"  Peak allocated: {mem:.1f} MB")
        print()


if __name__ == "__main__":
    main()
