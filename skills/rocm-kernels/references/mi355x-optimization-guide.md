# MI355X (gfx950) Optimization Guide

Deep dive into MI355X-specific optimizations for Triton kernels on ROCm.

## MI355X CDNA3+ Architecture

### Key Specifications

| Component | Value | vs MI300X |
|-----------|-------|-----------|
| Compute Capability | gfx950 | gfx942 |
| Architecture | CDNA3+ | CDNA3 |
| **XCDs (Chiplets)** | **32** | 8 |
| CUs Total | 256 | 228 |
| CUs per XCD | 8 | 28 |
| **LDS per CU** | **160 KB** | 64 KB |
| L2 Cache | 256 MB | 256 MB |
| Wavefront Size | 64 | 64 |
| GPU Memory | 288 GB HBM3e | 192 GB HBM3 |
| **Memory Bandwidth** | **8 TB/s** | 5.3 TB/s |
| FP16/BF16 Matrix TFLOPS | ~2500 | 1307 |
| FP8 Matrix TFLOPS | ~5000 | 2615 |
| MFMA Instructions | 16x16, 32x32 | 16x16, 32x32 |
| FP8 Format | float8_e4m3fn (OCP) | float8_e4m3fnuz (AMD) |

### Critical Architecture Differences from MI300X

1. **32 XCDs vs 8**: XCD swizzle must use `NUM_XCDS=32`
2. **8 CUs per XCD vs 28**: Finer-grained chiplet distribution
3. **160 KB LDS vs 64 KB**: 2.5x larger local memory per CU
4. **8 TB/s vs 5.3 TB/s**: 50% more memory bandwidth
5. **OCP FP8 vs AMD FP8**: Different FP8 format

## XCD Swizzle (MANDATORY for GEMM)

MI355X has 32 XCDs. Without proper swizzle, GEMM blocks cluster on a few XCDs, wasting 90%+ of the GPU.

### When to Use XCD Swizzle

| Kernel Type | XCD Swizzle? | Why |
|-------------|-------------|-----|
| GEMM / matmul | **YES, MANDATORY** | Multi-block work distribution |
| Elementwise | No | Single-block independent |
| Reduction | No | Row-independent |
| Normalization | No | Row-independent |
| Attention | **YES** (for Q@K and score@V) | Contains GEMM |

### XCD Swizzle Implementation

```python
NUM_XCDS = 32

@triton.jit
def gemm_with_xcd_swizzle(...):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pids = num_pid_m * num_pid_n

    # Step 1: XCD Swizzle
    pids_per_xcd = (num_pids + NUM_XCDS - 1) // NUM_XCDS
    xcd_id = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if local_pid < pids_per_xcd:
        remapped_pid = xcd_id * pids_per_xcd + local_pid
        if remapped_pid < num_pids:
            pid = remapped_pid

    # Step 2: L2 Cache Grouping (after XCD swizzle)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
```

### Performance Impact

| Config | Without XCD Swizzle | With XCD Swizzle | Improvement |
|--------|-------------------|-----------------|-------------|
| Square GEMM 4096x4096 | 0.3-0.5x | 0.8-1.2x | 2-4x |
| Tall-skinny GEMM | 0.4-0.6x | 0.7-1.0x | 1.5-2.5x |

## MFMA Instructions

Use 16x16 MFMA for optimal matrix core utilization:

```python
# Launch kernel with MFMA hint
kernel[grid](..., matrix_instr_nonkdim=16)
```

## LDS Optimization

MI355X has 160 KB LDS per CU—2.5x more than MI300X.

### LDS Budget Calculation

```
LDS usage = BLOCK_M × BLOCK_K × dtype_size + BLOCK_K × BLOCK_N × dtype_size
           × num_stages

Example (BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, FP16, num_stages=2):
  = (256×64×2 + 64×256×2) × 2 = 131,072 bytes = 128 KB < 160 KB ✓

Same config on MI300X (64 KB LDS):
  128 KB > 64 KB ✗ → Need num_stages=1 or smaller blocks
```

### Stage Configuration

| LDS Budget | MI355X num_stages | MI300X num_stages |
|------------|------------------|------------------|
| < 80 KB | 2-3 | 2 |
| 80-160 KB | 2 | 1 (or reduce blocks) |
| > 160 KB | 1 (or reduce blocks) | Not possible |

## Autotune Configurations

### Elementwise Operations

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=2),
    ],
    key=['n_elements'],
)
```

### GEMM Operations

```python
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
```

### Problem-Specific Block Sizes

| Problem Type | BLOCK_M | BLOCK_N | BLOCK_K | num_stages | num_warps | GROUP_M |
|-------------|---------|---------|---------|------------|-----------|---------|
| Square GEMM (M,N>=4096) | 256 | 256 | 32 | 3 | 8 | 16 |
| Large K (K > max(M,N)) | 128 | 128 | 64 | 2 | 8 | 8 |
| Fused GEMM+Activation | 128 | 128 | 64 | 2 | 8 | 8 |
| Element-wise ops | - | - | - | 2 | 4-16 | - |

## Precision and Numerical Stability

### FP32 Accumulation (Required)

```python
# Always accumulate in FP32
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(...):
    acc += tl.dot(a, b)
# Cast at store
c = acc.to(tl.float16)
```

### Math Operations

```python
# Cast to FP32 for transcendental functions
x_f32 = x.to(tl.float32)
result = tl.exp(x_f32)      # ✓
result = tl.log(x_f32)      # ✓
result = tl.sqrt(x_f32)     # ✓
result = 1.0 / x_f32        # ✓ (division in FP32)

# tanh workaround (tl.tanh not supported on AMD)
e2x = tl.exp(2.0 * x_f32)
tanh_x = (e2x - 1.0) / (e2x + 1.0)
```

## Performance Profiling

```bash
# Basic kernel profiling
rocprof --stats python your_kernel.py

# Detailed metrics
rocprofv3 -i metrics.txt python your_kernel.py

# Key metrics to watch:
# - L2 cache hit rate (target >70%)
# - VGPR usage (128+ may limit occupancy)
# - LDS usage (max 160 KB on MI355X)
# - Memory bandwidth utilization (target 40-60% of 8 TB/s)
```

## Environment Variables

```python
import os
# Block ping-pong for better latency hiding
os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
# Async memory copies
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'
```

## Best Practices Summary

1. **XCD Swizzle**: Always for GEMM, never for elementwise
2. **MFMA**: Use matrix_instr_nonkdim=16
3. **LDS**: Leverage 160 KB, but check with num_stages
4. **num_stages**: 2-3 (safe), up to 4 if LDS permits
5. **num_warps**: 8 is default, autotune 4-16
6. **BLOCK_SIZE**: Larger than MI300X (1024-16384 for 1D)
7. **GROUP_M**: 8 or 16 for L2 cache grouping
8. **FP32 acc**: Always accumulate in FP32
9. **Env vars**: Set BLOCK_PINGPONG and ASYNC_COPY
10. **Profile**: Use rocprof to validate optimizations
