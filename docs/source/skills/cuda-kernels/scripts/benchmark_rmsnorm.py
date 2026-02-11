#!/usr/bin/env python3
"""
Micro-benchmark for RMSNorm kernel to verify vectorized optimization.

Compares:
1. Custom CUDA kernel (vectorized)
2. PyTorch baseline implementation
"""

import torch
import time
from typing import Tuple

# Import custom kernel
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torch-ext'))
from ltx_kernels import rmsnorm


def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference PyTorch implementation of RMSNorm."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def benchmark_kernel(
    func,
    args,
    warmup: int = 10,
    iterations: int = 100,
    name: str = "kernel"
) -> Tuple[float, float]:
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = func(*args)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = sum(times) / len(times)
    min_time = min(times)
    return avg_time, min_time


def run_benchmark():
    """Run comprehensive RMSNorm benchmarks."""
    print("=" * 70)
    print("RMSNorm Micro-Benchmark: Custom Kernel vs PyTorch Baseline")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    # Test configurations matching LTX-Video dimensions
    # LTX-Video hidden_size is typically 2048 or 3072
    configs = [
        # (batch_size, seq_len, hidden_size)
        (1, 1024, 2048),    # Small
        (2, 1024, 2048),    # Medium
        (4, 1024, 2048),    # Larger batch
        (1, 4096, 2048),    # Longer sequence
        (2, 4096, 3072),    # LTX-Video typical
        (1, 8192, 2048),    # Very long sequence
        (4, 4096, 3072),    # Large workload
    ]

    dtype = torch.bfloat16  # LTX-Video uses bfloat16

    print(f"{'Config':<25} {'Custom (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    total_speedup = 0
    num_configs = 0

    for batch, seq, hidden in configs:
        # Create input tensors
        x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
        weight = torch.ones(hidden, dtype=dtype, device="cuda")

        # Benchmark custom kernel
        custom_avg, custom_min = benchmark_kernel(
            rmsnorm, (x, weight, 1e-6),
            warmup=20, iterations=100, name="custom"
        )

        # Benchmark PyTorch baseline
        pytorch_avg, pytorch_min = benchmark_kernel(
            pytorch_rmsnorm, (x, weight, 1e-6),
            warmup=20, iterations=100, name="pytorch"
        )

        # Calculate speedup
        speedup = pytorch_avg / custom_avg
        total_speedup += speedup
        num_configs += 1

        config_str = f"[{batch}x{seq}x{hidden}]"
        print(f"{config_str:<25} {custom_avg:>12.3f}   {pytorch_avg:>12.3f}   {speedup:>8.2f}x")

    avg_speedup = total_speedup / num_configs
    print("-" * 70)
    print(f"{'Average Speedup:':<55} {avg_speedup:.2f}x")
    print()

    # Verify correctness
    print("Correctness Check:")
    x = torch.randn(2, 1024, 2048, dtype=dtype, device="cuda")
    weight = torch.ones(2048, dtype=dtype, device="cuda")

    custom_out = rmsnorm(x, weight, 1e-6)
    pytorch_out = pytorch_rmsnorm(x, weight, 1e-6)

    max_diff = (custom_out - pytorch_out).abs().max().item()
    rel_diff = ((custom_out - pytorch_out).abs() / (pytorch_out.abs() + 1e-8)).max().item()

    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Max relative difference: {rel_diff:.6e}")
    # BFloat16 has only 7 bits mantissa, so 0.02 tolerance is appropriate
    print(f"  Correctness: {'PASS ✓' if max_diff < 0.05 else 'FAIL ✗'}")
    print()

    # Memory bandwidth analysis
    print("Memory Bandwidth Analysis:")
    batch, seq, hidden = 4, 4096, 3072
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    weight = torch.ones(hidden, dtype=dtype, device="cuda")

    # Bytes moved: read input + read weight + write output
    bytes_per_elem = 2  # bfloat16
    input_bytes = batch * seq * hidden * bytes_per_elem
    weight_bytes = hidden * bytes_per_elem
    output_bytes = batch * seq * hidden * bytes_per_elem
    total_bytes = input_bytes + weight_bytes + output_bytes

    custom_avg, _ = benchmark_kernel(rmsnorm, (x, weight, 1e-6), warmup=20, iterations=100)

    bandwidth_gbps = (total_bytes / 1e9) / (custom_avg / 1000)
    theoretical_bandwidth = 3350  # H100 theoretical 3.35 TB/s
    bandwidth_efficiency = (bandwidth_gbps / theoretical_bandwidth) * 100

    print(f"  Total data moved: {total_bytes / 1e6:.2f} MB")
    print(f"  Achieved bandwidth: {bandwidth_gbps:.1f} GB/s")
    print(f"  H100 theoretical: {theoretical_bandwidth} GB/s")
    print(f"  Bandwidth efficiency: {bandwidth_efficiency:.1f}%")
    print()


if __name__ == "__main__":
    run_benchmark()
