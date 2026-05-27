#!/usr/bin/env python3
"""
Micro-benchmark for all 4 Triton kernels on XPU: RMSNorm, RoPE 3D, GEGLU, AdaLN.

Measures:
  1. Correctness vs PyTorch reference
  2. Latency (custom vs baseline, warmup + averaged)
  3. Memory bandwidth utilization

Usage:
    python benchmark_kernels.py
    python benchmark_kernels.py --kernel rmsnorm
    python benchmark_kernels.py --kernel rope
    python benchmark_kernels.py --kernel geglu
    python benchmark_kernels.py --kernel adaln
    python benchmark_kernels.py --dtype float16

Requirements:
    python -m pip install -r scripts/requirements.txt
"""
import argparse
import time
from typing import Tuple

import torch
import triton
import triton.language as tl


# ============================================================================
# Kernel 1: RMSNorm
# ============================================================================
# CRITICAL: BLOCK_D must be >= D (hidden dimension).
# Using autotune with fixed BLOCK_D configs is WRONG because autotune may
# pick BLOCK_D < D, causing only partial row processing.
# Fix: compute BLOCK_D = next_power_of_2(D) dynamically in the Python wrapper.

@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, weight_ptr, out_ptr,
    stride_x, D,
    eps,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_D)
    mask = col_offsets < D

    x = tl.load(x_ptr + row * stride_x + col_offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / D
    rms_inv = tl.rsqrt(variance + eps)

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        result = x * rms_inv * w
    else:
        result = x * rms_inv

    tl.store(out_ptr + row * stride_x + col_offsets, result.to(x.dtype), mask=mask)


def triton_rmsnorm(x, weight=None, eps=1e-6):
    orig_shape = x.shape
    x_2d = x.contiguous().view(-1, x.shape[-1])
    out = torch.empty_like(x_2d)
    M, D = x_2d.shape
    has_weight = weight is not None
    if not has_weight:
        weight = torch.empty(0, device=x.device)

    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    rmsnorm_fwd_kernel[(M,)](
        x_2d, weight, out,
        x_2d.stride(0), D, float(eps), has_weight,
        BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=2,
    )
    return out.view(orig_shape)


def pytorch_rmsnorm(x, weight=None, eps=1e-6):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(variance + eps)
    if weight is not None:
        out = out * weight
    return out


# ============================================================================
# Kernel 2: RoPE 3D
# ============================================================================
# CRITICAL: cos/sin have shape [seq_len, head_dim], NOT [batch*seq_len, ...].
# When grid is (batch * seq_len, num_heads), we must use pid_s % seq_len
# to index into cos/sin to avoid out-of-bounds access for batch > 1.

@triton.jit
def rope_3d_fwd_kernel(
    qk_ptr, cos_ptr, sin_ptr, out_ptr,
    seq_len, num_heads, head_dim,
    stride_s, stride_h, stride_d,
    BLOCK_HD: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    half_dim = head_dim // 2
    offs = tl.arange(0, BLOCK_HD)
    mask = offs < half_dim

    base = pid_s * stride_s + pid_h * stride_h
    x0 = tl.load(qk_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(qk_ptr + base + half_dim + offs, mask=mask, other=0.0).to(tl.float32)

    seq_idx = pid_s % seq_len
    cos_val = tl.load(cos_ptr + seq_idx * head_dim + offs, mask=mask, other=1.0).to(tl.float32)
    sin_val = tl.load(sin_ptr + seq_idx * head_dim + offs, mask=mask, other=0.0).to(tl.float32)

    out0 = x0 * cos_val - x1 * sin_val
    out1 = x0 * sin_val + x1 * cos_val

    tl.store(out_ptr + base + offs, out0.to(x0.dtype), mask=mask)
    tl.store(out_ptr + base + half_dim + offs, out1.to(x0.dtype), mask=mask)


def triton_rope_3d(qk, cos, sin):
    qk = qk.contiguous()
    out = torch.empty_like(qk)
    batch, seq_len, num_heads, head_dim = qk.shape
    half_dim = head_dim // 2
    qk_flat = qk.view(batch * seq_len, num_heads, head_dim)
    out_flat = out.view(batch * seq_len, num_heads, head_dim)
    grid = (batch * seq_len, num_heads)
    BLOCK_HD = triton.next_power_of_2(half_dim)
    num_warps = 4 if BLOCK_HD <= 64 else 8
    rope_3d_fwd_kernel[grid](
        qk_flat, cos, sin, out_flat,
        seq_len, num_heads, head_dim,
        qk_flat.stride(0), qk_flat.stride(1), qk_flat.stride(2),
        BLOCK_HD=BLOCK_HD, num_warps=num_warps, num_stages=2,
    )
    return out


def pytorch_rope(qk, cos, sin):
    half = qk.shape[-1] // 2
    x0, x1 = qk[..., :half], qk[..., half:]
    cos_exp = cos.unsqueeze(0).unsqueeze(2)[:, :qk.shape[1], :, :half]
    sin_exp = sin.unsqueeze(0).unsqueeze(2)[:, :qk.shape[1], :, :half]
    out0 = x0 * cos_exp - x1 * sin_exp
    out1 = x0 * sin_exp + x1 * cos_exp
    return torch.cat([out0, out1], dim=-1)


# ============================================================================
# Kernel 3: GEGLU
# ============================================================================
# Same BLOCK_SIZE fix as RMSNorm: compute dynamically, do NOT autotune.

@triton.jit
def geglu_fwd_kernel(
    input_ptr, output_ptr,
    stride_in, stride_out, hidden_size,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < hidden_size

    gate = tl.load(input_ptr + row * stride_in + offs, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(input_ptr + row * stride_in + hidden_size + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Manual tanh for portability
    SQRT_2_OVER_PI = 0.7978845608028654
    tanh_arg = SQRT_2_OVER_PI * (gate + 0.044715 * gate * gate * gate)
    e2x = tl.exp(2.0 * tanh_arg)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)
    cdf = 0.5 * (1.0 + tanh_val)
    gelu_gate = gate * cdf
    result = gelu_gate * value

    tl.store(output_ptr + row * stride_out + offs, result.to(gate.dtype), mask=mask)


def triton_geglu(x):
    x = x.contiguous()
    *batch_dims, double_h = x.shape
    hidden_size = double_h // 2
    x_2d = x.view(-1, double_h)
    M = x_2d.shape[0]
    out = torch.empty(M, hidden_size, device=x.device, dtype=x.dtype)

    BLOCK_H = triton.next_power_of_2(hidden_size)
    num_warps = 4 if BLOCK_H <= 1024 else (8 if BLOCK_H <= 4096 else 16)
    geglu_fwd_kernel[(M,)](
        x_2d, out,
        x_2d.stride(0), out.stride(0), hidden_size,
        BLOCK_H=BLOCK_H, num_warps=num_warps, num_stages=2,
    )
    return out.view(*batch_dims, hidden_size)


def pytorch_geglu(x):
    hidden_size = x.shape[-1] // 2
    gate, value = x[..., :hidden_size], x[..., hidden_size:]
    return torch.nn.functional.gelu(gate, approximate='tanh') * value


# ============================================================================
# Kernel 4: AdaLN
# ============================================================================
# Same BLOCK_D fix: compute dynamically.

@triton.jit
def adaln_fwd_kernel(
    x_ptr, weight_ptr, scale_ptr, shift_ptr, out_ptr,
    stride_x, stride_cond, D,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / D
    rms_inv = tl.rsqrt(variance + eps)
    x_norm = x * rms_inv

    w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    scale = tl.load(scale_ptr + row * stride_cond + offs, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + row * stride_cond + offs, mask=mask, other=0.0).to(tl.float32)

    out = x_norm * w * (1.0 + scale) + shift
    tl.store(out_ptr + row * stride_x + offs, out.to(x.dtype), mask=mask)


def triton_adaln(x, weight, scale, shift, eps=1e-6):
    x_flat = x.contiguous().view(-1, x.shape[-1])
    scale_flat = scale.contiguous().view(-1, x.shape[-1])
    shift_flat = shift.contiguous().view(-1, x.shape[-1])
    out = torch.empty_like(x_flat)
    M, D = x_flat.shape

    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    adaln_fwd_kernel[(M,)](
        x_flat, weight, scale_flat, shift_flat, out,
        x_flat.stride(0), scale_flat.stride(0), D, float(eps),
        BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=2,
    )
    return out.view_as(x)


def pytorch_adaln(x, weight, scale, shift, eps=1e-6):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight * (1.0 + scale) + shift


# ============================================================================
# Benchmark Utilities
# ============================================================================

def benchmark_fn(func, args, warmup=20, iterations=100) -> Tuple[float, float]:
    for _ in range(warmup):
        func(*args)
    torch.xpu.synchronize()

    times = []
    for _ in range(iterations):
        torch.xpu.synchronize()
        start = time.perf_counter()
        func(*args)
        torch.xpu.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times), min(times)


def check_correctness(out, ref, name, dtype):
    max_abs = (out.float() - ref.float()).abs().max().item()
    max_rel = ((out.float() - ref.float()).abs() / (ref.float().abs() + 1e-8)).max().item()

    # BF16 has 7-bit mantissa; for values ~8-16 the ULP is 0.0625-0.125
    # FP16 has 10-bit mantissa; tighter but RoPE trig ops can accumulate 1-2 ULP error
    atol = 0.15 if dtype == torch.bfloat16 else 0.02
    passed = max_abs < atol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}")
    return passed


# ============================================================================
# Benchmark Runners
# ============================================================================

def benchmark_rmsnorm(dtype):
    print("\n" + "=" * 70)
    print("BENCHMARK: RMSNorm (168 instances in LTX-Video)")
    print("=" * 70)

    configs = [
        (1, 1024, 2048),
        (2, 1024, 2048),
        (4, 1024, 2048),
        (1, 4096, 2048),
        (2, 4096, 3072),
        (1, 8192, 2048),
        (4, 4096, 3072),
    ]

    print(f"\n{'Config':<25} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    all_correct = True
    total_speedup = 0

    for batch, seq, hidden in configs:
        x = torch.randn(batch, seq, hidden, dtype=dtype, device="xpu")
        w = torch.ones(hidden, dtype=dtype, device="xpu")

        ref = pytorch_rmsnorm(x, w)
        out = triton_rmsnorm(x, w)
        if not check_correctness(out, ref, f"[{batch}x{seq}x{hidden}]", dtype):
            all_correct = False

        t_avg, _ = benchmark_fn(triton_rmsnorm, (x, w))
        p_avg, _ = benchmark_fn(pytorch_rmsnorm, (x, w))
        speedup = p_avg / t_avg
        total_speedup += speedup

        print(f"  [{batch}x{seq}x{hidden}]{'':<13} {t_avg:>10.3f}     {p_avg:>10.3f}     {speedup:>7.2f}x")

    # No-weight variant
    print("\n  -- No-weight variant (elementwise_affine=False) --")
    x = torch.randn(2, 4096, 2048, dtype=dtype, device="xpu")
    ref_nw = pytorch_rmsnorm(x, None)
    out_nw = triton_rmsnorm(x, None)
    check_correctness(out_nw, ref_nw, "no-weight [2x4096x2048]", dtype)

    avg_speedup = total_speedup / len(configs)
    print(f"\n  Average speedup: {avg_speedup:.2f}x")

    # Bandwidth analysis
    batch, seq, hidden = 4, 4096, 3072
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="xpu")
    w = torch.ones(hidden, dtype=dtype, device="xpu")
    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total_bytes = batch * seq * hidden * bytes_per_elem * 2 + hidden * bytes_per_elem
    t_avg, _ = benchmark_fn(triton_rmsnorm, (x, w))
    bw_gbps = (total_bytes / 1e9) / (t_avg / 1000)
    print(f"\n  Bandwidth analysis [{batch}x{seq}x{hidden}]:")
    print(f"    Data moved: {total_bytes / 1e6:.2f} MB")
    print(f"    Achieved: {bw_gbps:.1f} GB/s")

    return all_correct, avg_speedup


def benchmark_rope(dtype):
    print("\n" + "=" * 70)
    print("BENCHMARK: RoPE 3D (Video Position Encoding)")
    print("=" * 70)

    configs = [
        (1, 1024, 16, 64),
        (1, 4096, 16, 64),
        (2, 4096, 16, 128),
        (1, 8192, 32, 64),
    ]

    print(f"\n{'Config':<30} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 75)

    all_correct = True
    total_speedup = 0

    for batch, seq, heads, hdim in configs:
        qk = torch.randn(batch, seq, heads, hdim, dtype=dtype, device="xpu")
        cos = torch.randn(seq, hdim, dtype=dtype, device="xpu")
        sin = torch.randn(seq, hdim, dtype=dtype, device="xpu")

        ref = pytorch_rope(qk, cos, sin)
        out = triton_rope_3d(qk, cos, sin)
        if not check_correctness(out, ref, f"[{batch}x{seq}x{heads}x{hdim}]", dtype):
            all_correct = False

        t_avg, _ = benchmark_fn(triton_rope_3d, (qk, cos, sin))
        p_avg, _ = benchmark_fn(pytorch_rope, (qk, cos, sin))
        speedup = p_avg / t_avg
        total_speedup += speedup

        cfg = f"[{batch}x{seq}x{heads}x{hdim}]"
        print(f"  {cfg:<28} {t_avg:>10.3f}     {p_avg:>10.3f}     {speedup:>7.2f}x")

    avg_speedup = total_speedup / len(configs)
    print(f"\n  Average speedup: {avg_speedup:.2f}x")
    return all_correct, avg_speedup


def benchmark_geglu(dtype):
    print("\n" + "=" * 70)
    print("BENCHMARK: GEGLU (For SD3/FLUX, NOT LTX-Video)")
    print("=" * 70)

    configs = [
        (1, 1024, 2048),
        (2, 1024, 4096),
        (2, 4096, 3072),
        (4, 4096, 4096),
    ]

    print(f"\n{'Config':<30} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 75)

    all_correct = True
    total_speedup = 0

    for batch, seq, hidden in configs:
        x = torch.randn(batch, seq, hidden * 2, dtype=dtype, device="xpu")

        ref = pytorch_geglu(x)
        out = triton_geglu(x)
        if not check_correctness(out, ref, f"[{batch}x{seq}x{hidden*2}]", dtype):
            all_correct = False

        t_avg, _ = benchmark_fn(triton_geglu, (x,))
        p_avg, _ = benchmark_fn(pytorch_geglu, (x,))
        speedup = p_avg / t_avg
        total_speedup += speedup

        cfg = f"[{batch}x{seq}x{hidden*2}->{hidden}]"
        print(f"  {cfg:<28} {t_avg:>10.3f}     {p_avg:>10.3f}     {speedup:>7.2f}x")

    avg_speedup = total_speedup / len(configs)
    print(f"\n  Average speedup: {avg_speedup:.2f}x")
    return all_correct, avg_speedup


def benchmark_adaln(dtype):
    print("\n" + "=" * 70)
    print("BENCHMARK: AdaLN (Fused Norm + Conditioning for DiT)")
    print("=" * 70)

    configs = [
        (1, 1024, 2048),
        (2, 1024, 2048),
        (2, 4096, 3072),
        (4, 4096, 3072),
    ]

    print(f"\n{'Config':<25} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    all_correct = True
    total_speedup = 0

    for batch, seq, hidden in configs:
        x = torch.randn(batch, seq, hidden, dtype=dtype, device="xpu")
        w = torch.ones(hidden, dtype=dtype, device="xpu")
        scale = torch.randn(batch, seq, hidden, dtype=dtype, device="xpu") * 0.1
        shift = torch.randn(batch, seq, hidden, dtype=dtype, device="xpu") * 0.1

        ref = pytorch_adaln(x, w, scale, shift)
        out = triton_adaln(x, w, scale, shift)
        if not check_correctness(out, ref, f"[{batch}x{seq}x{hidden}]", dtype):
            all_correct = False

        t_avg, _ = benchmark_fn(triton_adaln, (x, w, scale, shift))
        p_avg, _ = benchmark_fn(pytorch_adaln, (x, w, scale, shift))
        speedup = p_avg / t_avg
        total_speedup += speedup

        print(f"  [{batch}x{seq}x{hidden}]{'':<13} {t_avg:>10.3f}     {p_avg:>10.3f}     {speedup:>7.2f}x")

    avg_speedup = total_speedup / len(configs)
    print(f"\n  Average speedup: {avg_speedup:.2f}x")
    return all_correct, avg_speedup


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton kernels on XPU")
    parser.add_argument("--kernel", type=str, default="all",
                        choices=["all", "rmsnorm", "rope", "geglu", "adaln"])
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=" * 70)
    print("XPU Triton Kernel Micro-Benchmark")
    print("=" * 70)
    print(f"Device: {torch.xpu.get_device_name(0)}")
    print(f"Dtype:  {dtype}")

    results = {}
    runners = {
        "rmsnorm": benchmark_rmsnorm,
        "rope": benchmark_rope,
        "geglu": benchmark_geglu,
        "adaln": benchmark_adaln,
    }

    if args.kernel == "all":
        for name, runner in runners.items():
            correct, speedup = runner(dtype)
            results[name] = {"correct": correct, "speedup": speedup}
    else:
        correct, speedup = runners[args.kernel](dtype)
        results[args.kernel] = {"correct": correct, "speedup": speedup}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Kernel':<15} {'Correct':<12} {'Avg Speedup':<15}")
    print("-" * 42)
    for name, r in results.items():
        status = "PASS" if r["correct"] else "FAIL"
        print(f"{name:<15} {status:<12} {r['speedup']:.2f}x")

    all_pass = all(r["correct"] for r in results.values())
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
