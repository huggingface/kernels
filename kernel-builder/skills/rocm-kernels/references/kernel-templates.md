# Triton Kernel Templates for ROCm (LTX-Video Operators)

Copy-paste ready Triton kernel templates for RMSNorm, RoPE 3D, GEGLU, and AdaLN on AMD GPUs.

## Required Header

**Every kernel file MUST start with:**

```python
import os
os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'

import torch
import torch.nn as nn
import triton
import triton.language as tl
```

## Template 1: RMSNorm (Core Target)

Row-wise reduction. **168 instances** in LTX-Video. Handles both with-weight and no-weight variants.

**CRITICAL: Do NOT autotune BLOCK_D.** Autotune may select `BLOCK_D < D`, causing partial row processing and completely wrong results. Always compute `BLOCK_D = triton.next_power_of_2(D)` dynamically.

### Triton Kernel

```python
@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, weight_ptr, out_ptr,
    stride_x, D,
    eps: tl.constexpr,
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
```

### Python API

```python
def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
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
        x_2d.stride(0), D, eps, has_weight,
        BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=2,
    )
    return out.view(orig_shape)
```

### Benchmark

```python
def benchmark_rmsnorm():
    configs = [
        (1, 1024, 2048),
        (2, 1024, 2048),
        (2, 4096, 3072),
    ]
    for batch, seq, hidden in configs:
        x = torch.randn(batch, seq, hidden, device='cuda', dtype=torch.float16)
        w = torch.ones(hidden, device='cuda', dtype=torch.float16)

        # Reference
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

        # Custom
        out = triton_rmsnorm(x, w, eps=1e-6)

        # Verify
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)
        print(f"[{batch}x{seq}x{hidden}] ✓ Correct")
```

## Template 2: RoPE 3D (Video Position Encoding)

Element-wise rotation. Splits head_dim into temporal + spatial (height + width) components.

**CRITICAL: cos/sin have shape `[seq_len, head_dim]`, NOT `[batch*seq_len, ...]`.** When the grid flattens the batch dimension, use `pid_s % seq_len` to index cos/sin, otherwise batch > 1 causes out-of-bounds GPU crash.

### Triton Kernel

```python
@triton.jit
def rope_3d_fwd_kernel(
    qk_ptr, cos_ptr, sin_ptr, out_ptr,
    seq_len, num_heads, head_dim,
    stride_s, stride_h, stride_d,
    BLOCK_HD: tl.constexpr,
):
    pid_s = tl.program_id(0)  # ranges [0, batch * seq_len)
    pid_h = tl.program_id(1)

    half_dim = head_dim // 2
    offs = tl.arange(0, BLOCK_HD)
    mask = offs < half_dim

    base = pid_s * stride_s + pid_h * stride_h
    x0 = tl.load(qk_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(qk_ptr + base + half_dim + offs, mask=mask, other=0.0).to(tl.float32)

    seq_idx = pid_s % seq_len  # wrap for batch > 1
    cos_val = tl.load(cos_ptr + seq_idx * head_dim + offs, mask=mask, other=1.0).to(tl.float32)
    sin_val = tl.load(sin_ptr + seq_idx * head_dim + offs, mask=mask, other=0.0).to(tl.float32)

    out0 = x0 * cos_val - x1 * sin_val
    out1 = x0 * sin_val + x1 * cos_val

    tl.store(out_ptr + base + offs, out0.to(x0.dtype), mask=mask)
    tl.store(out_ptr + base + half_dim + offs, out1.to(x0.dtype), mask=mask)
```

### Python API

```python
def triton_rope_3d(
    qk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply 3D RoPE to Q or K tensor.

    Args:
        qk: [batch, seq_len, num_heads, head_dim]
        cos: [seq_len, head_dim]  — NOT batch-expanded!
        sin: [seq_len, head_dim]
    """
    qk = qk.contiguous()
    out = torch.empty_like(qk)
    batch, seq_len, num_heads, head_dim = qk.shape

    qk_flat = qk.view(batch * seq_len, num_heads, head_dim)
    out_flat = out.view(batch * seq_len, num_heads, head_dim)

    BLOCK_HD = triton.next_power_of_2(head_dim // 2)
    num_warps = 4 if BLOCK_HD <= 64 else 8

    rope_3d_fwd_kernel[(batch * seq_len, num_heads)](
        qk_flat, cos, sin, out_flat,
        seq_len, num_heads, head_dim,
        qk_flat.stride(0), qk_flat.stride(1), qk_flat.stride(2),
        BLOCK_HD=BLOCK_HD, num_warps=num_warps, num_stages=2,
    )
    return out
```

## Template 3: GEGLU (For SD3/FLUX)

Gated activation: `GELU(gate) * value`. Input splits in half along last dim.

**Note: LTX-Video uses GELU, NOT GEGLU. This template is for SD3/FLUX.**

### Triton Kernel

```python
@triton.jit
def geglu_fwd_kernel(
    input_ptr, output_ptr,
    stride_in, stride_out, hidden_size,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < hidden_size

    gate = tl.load(input_ptr + row * stride_in + offs,
                   mask=mask, other=0.0).to(tl.float32)
    value = tl.load(input_ptr + row * stride_in + hidden_size + offs,
                    mask=mask, other=0.0).to(tl.float32)

    # Manual tanh — tl.math.tanh / tl.libdevice.tanh NOT available on ROCm
    SQRT_2_OVER_PI = 0.7978845608028654
    tanh_arg = SQRT_2_OVER_PI * (gate + 0.044715 * gate * gate * gate)
    e2x = tl.exp(2.0 * tanh_arg)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)
    cdf = 0.5 * (1.0 + tanh_val)
    gelu_gate = gate * cdf

    result = gelu_gate * value
    tl.store(output_ptr + row * stride_out + offs, result.to(gate.dtype), mask=mask)
```

### Python API

```python
def triton_geglu(x: torch.Tensor) -> torch.Tensor:
    """
    GEGLU activation: GELU(x[..., :H]) * x[..., H:]

    Input: [..., 2*hidden_size] → Output: [..., hidden_size]
    """
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
```

## Template 4: AdaLN (Adaptive Layer Normalization)

Fused RMSNorm + adaptive conditioning for DiT blocks.
Formula: `norm(x) * weight * (1 + scale) + shift`

### Triton Kernel

```python
@triton.jit
def adaln_fwd_kernel(
    x_ptr, weight_ptr, scale_ptr, shift_ptr, out_ptr,
    stride_x, stride_cond, D,
    eps: tl.constexpr,
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
```

### Python API

```python
def triton_adaln(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Adaptive Layer Normalization for DiT blocks.

    Args:
        x: [batch, seq, hidden]
        weight: [hidden]
        scale: [batch, seq, hidden] or [batch, 1, hidden]
        shift: [batch, seq, hidden] or [batch, 1, hidden]
    """
    x_flat = x.contiguous().view(-1, x.shape[-1])
    scale_flat = scale.contiguous().view(-1, x.shape[-1])
    shift_flat = shift.contiguous().view(-1, x.shape[-1])
    out = torch.empty_like(x_flat)
    M, D = x_flat.shape

    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    adaln_fwd_kernel[(M,)](
        x_flat, weight, scale_flat, shift_flat, out,
        x_flat.stride(0), scale_flat.stride(0), D, eps,
        BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=2,
    )
    return out.view_as(x)
```

## Common Math Replacements for ROCm

| Standard | ROCm Triton Replacement |
|----------|------------------------|
| `tl.tanh(x)` | Manual: `e2x = tl.exp(2.0*x); (e2x-1)/(e2x+1)` |
| `tl.math.tanh(x)` | **Also NOT available on ROCm** — use manual formula above |
| `tl.libdevice.*` | Remove entirely, use manual implementations |
| `min(a, b)` | `tl.minimum(a, b)` |
| `max(a, b)` | `tl.maximum(a, b)` |
| GELU exact | `0.5 * x * (1 + erf(x / sqrt(2)))` |
| GELU approx | `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))` |

## Kernel-Specific Guidelines

### RMSNorm
- Input: `[..., hidden_size]` — flatten to 2D `[M, D]`
- Epsilon default: 1e-6
- **Weight may be None** if `elementwise_affine=False`
- Always accumulate `x*x` sum in FP32
- **BLOCK_D = `triton.next_power_of_2(D)`** — compute in wrapper, NEVER autotune
- Autotuning BLOCK_D is dangerous: if BLOCK_D < D, only partial row is processed → wrong results

### RoPE 3D
- 1D: `[batch, seq, heads, head_dim]` for text
- 3D: `[batch, t*h*w, heads, head_dim]` for video
- LTX-Video computes RoPE via `LTXVideoRotaryPosEmbed` — kernel replaces the apply step
- head_dim typically 64 or 128
- **cos/sin shape is `[seq_len, head_dim]`** — use `pid_s % seq_len` for batch > 1

### GEGLU vs GELU
- **GEGLU**: Input `[B, S, 2*H]` → Output `[B, S, H]` — gate/value split
- **GELU**: Standard activation, no split
- **LTX-Video uses GELU, NOT GEGLU**
- GEGLU is for SD3/FLUX

### AdaLN
- Formula: `norm(x) * weight * (1 + scale) + shift`
- Scale/shift come from timestep embedding MLP
- DiT computes 6 values per block: `(scale1, shift1, gate1, scale2, shift2, gate2)`
- Fusing norm + conditioning saves one memory round-trip

## Template 5: GEMM with XCD Swizzle (MI355X)

Tiled matrix multiplication with XCD swizzle for MI355X (32 XCDs). **Mandatory** for any GEMM-like operation on MI355X — without it, work clusters on a few chiplets, wasting 90%+ of the GPU.

> See [mi355x-optimization-guide.md](mi355x-optimization-guide.md) for architecture details.

**When to use XCD swizzle:** GEMM, batched GEMM, attention (Q@K, score@V). NOT needed for elementwise, reduction, or normalization kernels.

### Triton Kernel

```python
NUM_XCDS = 32  # MI355X has 32 XCDs

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8},
                      num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
                      num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8},
                      num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8},
                      num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_xcd_swizzle_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pids = num_pid_m * num_pid_n

    # --- XCD Swizzle: distribute blocks across 32 chiplets ---
    pids_per_xcd = (num_pids + NUM_XCDS - 1) // NUM_XCDS
    xcd_id = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if local_pid < pids_per_xcd:
        remapped_pid = xcd_id * pids_per_xcd + local_pid
        if remapped_pid < num_pids:
            pid = remapped_pid

    # --- L2 Cache Grouping (after XCD swizzle) ---
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --- Compute GEMM tile ---
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # --- Store result ---
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)
```

### Python API

```python
def triton_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication C = A @ B with XCD swizzle for MI355X.

    Args:
        a: [M, K] input matrix
        b: [K, N] input matrix
    Returns:
        c: [M, N] output matrix
    """
    assert a.shape[1] == b.shape[0], "Inner dimensions must match"
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_xcd_swizzle_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
```

### Benchmark

```python
def benchmark_gemm():
    configs = [(4096, 4096, 4096), (8192, 8192, 4096), (2048, 8192, 2048)]
    for M, N, K in configs:
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)

        ref = torch.mm(a, b)
        out = triton_gemm(a, b)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-1)

        # Benchmark
        for _ in range(10):
            triton_gemm(a, b)
            torch.mm(a, b)
        torch.cuda.synchronize()

        import time
        iters = 50
        start = time.perf_counter()
        for _ in range(iters):
            triton_gemm(a, b)
        torch.cuda.synchronize()
        custom_ms = (time.perf_counter() - start) / iters * 1000

        start = time.perf_counter()
        for _ in range(iters):
            torch.mm(a, b)
        torch.cuda.synchronize()
        torch_ms = (time.perf_counter() - start) / iters * 1000

        print(f"[{M}x{N}x{K}] Custom: {custom_ms:.2f}ms, Torch: {torch_ms:.2f}ms, "
              f"Speedup: {torch_ms/custom_ms:.2f}x")
```

### GEMM-Specific Guidelines

- **XCD Swizzle is MANDATORY** on MI355X for any GEMM — without it, expect 0.3-0.5x
- **L2 Cache Grouping** (`GROUP_M=8-16`): Improves L2 hit rate after XCD swizzle
- **MFMA**: Use `matrix_instr_nonkdim=16` for MI355X matrix cores
- **FP32 accumulation**: Always accumulate in FP32, cast at store
- **LDS budget**: Check `BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N` * dtype * num_stages < 160 KB
- **Autotune**: GEMM benefits heavily from autotuning — always include 4+ configs
- **R9700**: Does NOT have XCDs — remove the XCD swizzle section for RDNA4
