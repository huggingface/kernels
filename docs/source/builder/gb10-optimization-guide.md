# GB10 (Blackwell, sm_121) Optimization Guide

Hardware-specific guidance for writing optimized CUDA kernels targeting the
NVIDIA GB10 GPU found in the DGX Spark desktop system. The GB10 is the first
consumer/workstation Blackwell chip and has a fundamentally different memory
architecture from data-center GPUs — understanding the differences is key to
writing fast kernels.

## Architecture overview

### Key specifications

| Component | Specification | Notes |
|-----------|---------------|-------|
| Compute Capability | 12.1 (sm_121) | Binary-compatible with sm_120 |
| SMs | 48 | Fewer than A100 (108) / H100 (132) |
| CUDA Cores | 6,144 | 128 per SM |
| Tensor Cores | 192 | 5th gen Blackwell, FP4/FP8/FP16/BF16 |
| L2 Cache | 24 MB | Smaller than H100 (50 MB) |
| Shared Memory | 100 KB/SM | Opt-in max: 99 KB/block |
| Registers | 64K 32-bit/SM | 65,536 per SM |
| Memory | 128 GB unified (LPDDR5X) | Shared with Grace CPU |
| Memory Bus | 256-bit @ 8533 MHz | ~273 GB/s theoretical per direction |
| Max Threads/SM | 1,536 | 48 warps (vs 64 on H100) |
| Max Threads/Block | 1,024 | 32 warps |
| Warp Size | 32 | Unchanged |
| GPU Clock | 2.4 GHz (boost 3.0 GHz) | |

### Blackwell features available on GB10

- **FP4 / FP8 tensor core support** — native 4-bit and 8-bit floating point
- **sm_120/sm_121 binary compatibility** — code compiled for compute_120 runs on sm_121
- **TMA (Tensor Memory Accelerator)** — hardware bulk memory copy
- **Thread Block Clusters** — cooperative groups of thread blocks
- **128 GB unified memory** — no PCIe transfers, CPU and GPU share the same physical memory

### What makes GB10 different

The GB10 is an **integrated GPU** — the Grace CPU and Blackwell GPU share a
single 128 GB LPDDR5X memory pool. This has major implications:

1. **No PCIe bottleneck.** CPU↔GPU data transfers are coherent memory
   accesses, not DMA copies. `cudaMemcpy` is effectively free for small
   transfers.
2. **Lower raw bandwidth than HBM.** The GB10's ~273 GB/s theoretical
   bandwidth is roughly 8× lower than the H100's 3.35 TB/s HBM3. Bandwidth
   efficiency matters even more here.
3. **Huge effective VRAM.** The full 128 GB is available to CUDA. Models that
   need offloading on 80 GB H100s fit entirely in memory on GB10.
4. **No swap recommended.** Unified memory + Linux OOM can cause zombie
   processes. Run `sudo swapoff -a` before long GPU workloads.

## Comparison with H100 and A100

| Spec | GB10 | H100 | A100 |
|------|------|------|------|
| SMs | 48 | 132 | 108 |
| Threads/SM | 1,536 | 2,048 | 2,048 |
| Warps/SM | 48 | 64 | 64 |
| Shared Mem/SM | 100 KB | 192 KB | 164 KB |
| L2 Cache | 24 MB | 50 MB | 40 MB |
| Memory BW | ~273 GB/s | 3,350 GB/s | 2,039 GB/s |
| Memory | 128 GB unified | 80 GB HBM3 | 80 GB HBM2e |
| BF16 Tensor | Yes | Yes | Yes |
| FP8 Tensor | Yes | Yes | No |
| FP4 Tensor | Yes | No | No |
| Compute Cap | sm_121 | sm_90 | sm_80 |

### Practical impact

Kernels that are **memory-bound** (normalization, element-wise ops, small
reductions) will see proportionally lower throughput on GB10 compared to H100
because of the bandwidth gap. Kernels that are **compute-bound** (large
matmuls, convolutions) will be limited by the 48 SMs but can still achieve
high tensor core utilization.

The sweet spot for custom kernels on GB10 is **fused operations** — reducing
memory round-trips matters even more when bandwidth is the bottleneck.

## Measured performance

All benchmarks run on NVIDIA GB10, CUDA 13.0, PyTorch 2.10, BFloat16.

### Compute throughput

| Operation | Measured | Notes |
|-----------|----------|-------|
| FP32 MatMul (4096×4096) | 16.4 TFLOPS | CUDA cores only |
| BF16 MatMul (4096×4096) | 91.1 TFLOPS | Tensor cores |
| BF16 MatMul (2048×2048) | 92.3 TFLOPS | Peak observed |

### Memory bandwidth

| Test | Measured | Efficiency |
|------|----------|------------|
| BF16 copy (256 MB) | 218 GB/s | 40% of theoretical |
| FP32 copy (256 MB) | 219 GB/s | 40% of theoretical |
| RMSNorm (vectorized) | 185 GB/s | 34% of theoretical |

The ~40% measured efficiency is expected for LPDDR5X — unlike HBM, there is
no wide internal bus, and the memory controller must share bandwidth with the
CPU.

### RMSNorm kernel benchmarks

Vectorized custom kernel vs PyTorch baseline (`x.pow(2).mean().rsqrt() * w`):

| Shape | Custom (ms) | PyTorch (ms) | Speedup |
|:---|:---:|:---:|:---:|
| [1×1024×2048] | 0.034 | 0.051 | **1.51×** |
| [2×1024×2048] | 0.063 | 0.154 | **2.44×** |
| [4×1024×2048] | 0.161 | 0.415 | **2.57×** |
| [1×4096×2048] | 0.158 | 0.441 | **2.78×** |
| [2×4096×3072] | 0.537 | 1.583 | **2.95×** |
| [1×8192×2048] | 0.356 | 1.013 | **2.84×** |
| [4×4096×3072] | 1.061 | 3.187 | **3.00×** |
| **Average** | | | **2.59×** |

The speedup comes from vectorized loads (`__nv_bfloat162`) and fused
reduction — PyTorch's decomposed `pow → mean → rsqrt → mul` touches memory
4× per element while the custom kernel touches it twice (read + write).

### RMSNorm scaling by hidden size

| Hidden Size | Custom (ms) | Speedup | Notes |
|:-----------:|:-----------:|:-------:|:------|
| 512 | 0.008 | 3.37× | Small — kernel launch overhead dominates PyTorch |
| 1024 | 0.010 | 2.87× | |
| 2048 | 0.014 | 2.05× | Common LLM dimension |
| 4096 | 0.016 | 2.21× | LLaMA-class models |
| 8192 | 0.018 | 3.22× | Large models, vectorization shines |

## Memory hierarchy optimization

### Unified memory behavior

The GB10's unified memory means there is **no separate device memory
allocation** — `cudaMalloc` carves out pages from the same physical LPDDR5X
pool that the CPU uses. Implications:

```python
# These are effectively free on GB10 (no DMA, just page table update):
tensor = torch.randn(1000, 1000).cuda()  # Near-instant
tensor_cpu = tensor.cpu()                 # Near-instant

# But large contiguous allocations can still fragment:
# Prefer model loading to GPU upfront rather than repeated small transfers
```

For kernel authors, unified memory means:
- **Pinned memory is unnecessary.** `cudaHostAlloc` offers no benefit since
  all memory is already coherent.
- **Zero-copy CPU access is possible** but slow — GPU cache coherence
  protocol adds latency. Always copy to a GPU tensor first.
- **cudaMemcpyAsync is still useful** for overlapping compute with data
  movement to L2/shared memory.

### Vectorized memory access (critical)

With lower bandwidth, maximizing bytes per transaction is essential:

```cuda
// BF16: 2 elements per 32-bit load via __nv_bfloat162
const __nv_bfloat162* vec_input = reinterpret_cast<const __nv_bfloat162*>(row);
__nv_bfloat162 v = vec_input[i];
float v0 = __bfloat162float(v.x);
float v1 = __bfloat162float(v.y);

// FP16: 2 elements per 32-bit load via __half2
const __half2* vec_input = reinterpret_cast<const __half2*>(row);

// FP32: 4 elements per 128-bit load via float4
const float4* vec_input = reinterpret_cast<const float4*>(row);
```

Always use `#pragma unroll 4` on vectorized loops — the compiler can
schedule loads to hide latency.

### L2 cache (24 MB)

Smaller than H100's 50 MB. Tile sizes for attention should be adjusted:

```
H100: BLOCK_M=128, BLOCK_N=64, head_dim=64  →  16 KB/tile  →  ~3000 tiles in L2
GB10: BLOCK_M=64,  BLOCK_N=64, head_dim=64  →   8 KB/tile  →  ~3000 tiles in L2
```

Use smaller `BLOCK_M` to keep the working set in L2. The reduced SM count
(48 vs 132) means fewer concurrent tiles anyway.

### Shared memory (100 KB/SM)

The GB10 has 100 KB of shared memory per SM, with up to 99 KB available per
block via opt-in:

```cuda
// Request max shared memory for a block
cudaFuncSetAttribute(
    my_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    99 * 1024  // 99 KB opt-in max
);
```

Default allocation is 48 KB/block. For reduction kernels that only need a
few floats of shared memory, the default is fine. Request more only for
tiled matmul or attention kernels.

**Bank conflicts** still apply — 32 banks, 4 bytes per bank. Same avoidance
strategies as previous architectures (padding, sequential access).

## Occupancy tuning

### Warps and blocks per SM

The GB10 supports **48 warps (1,536 threads) per SM** — fewer than the 64
warps on H100/A100. This changes the occupancy math:

| Threads/Block | Warps/Block | Max Blocks/SM | Active Warps | Occupancy |
|:---:|:---:|:---:|:---:|:---:|
| 128 | 4 | 12 | 48 | 100% |
| 256 | 8 | 6 | 48 | 100% |
| 512 | 16 | 3 | 48 | 100% |
| 1024 | 32 | 1 | 32 | 67% |

**Key insight:** Using 1024 threads/block on GB10 means only 1 block per SM
and 67% occupancy. Prefer 256–512 threads/block to maintain 100% occupancy.

For reduction kernels (RMSNorm, LayerNorm), 256 threads is optimal:
```cuda
// Good: 256 threads → 6 blocks/SM → 100% occupancy
dim3 block(256);

// Risky: 1024 threads → 1 block/SM → 67% occupancy, less latency hiding
dim3 block(1024);
```

### Grid sizing

With 48 SMs, aim for grid sizes that are multiples of 48 for full GPU
utilization:

```cuda
// Good: 48, 96, 144, 192, ... blocks
// Acceptable: anything ≥ 48 (tail effects are minor)
// Bad: 1–47 blocks (some SMs idle)

const int num_blocks = (num_rows + 0) * 1;  // Naturally ≥ 48 for real workloads
```

For small workloads (< 48 rows), some SMs will be idle. Consider batching
or processing multiple small tensors in a single kernel launch.

## Precision and numerical stability

### Supported precisions

| Type | Tensor Core | CUDA Core | Notes |
|------|:-----------:|:---------:|-------|
| FP4 | Yes | No | New in Blackwell |
| FP8 (E4M3/E5M2) | Yes | No | Training + inference |
| FP16 | Yes | Yes | Inference standard |
| BF16 | Yes | Yes | Training standard |
| TF32 | Yes | No | Default for FP32 matmul |
| FP32 | — | Yes | Accumulation, reductions |

### Type conversion helpers

PyTorch compiles with `-D__CUDA_NO_HALF_OPERATORS__`, so implicit conversions
are disabled. Every kernel file must include explicit helpers:

```cuda
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) { return x; }
template <>
__device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}
```

**Do not use `static_cast<__half>(float_val)`** — it will fail to compile
with the CUDA half operators disabled.

### Mixed precision pattern

Always accumulate reductions in FP32:

```cuda
float sum_sq = 0.0f;  // FP32 accumulator
for (int i = tid; i < hidden_size / 2; i += stride) {
    __nv_bfloat162 v = vec_input[i];
    float v0 = __bfloat162float(v.x);
    float v1 = __bfloat162float(v.y);
    sum_sq += v0 * v0 + v1 * v1;  // Accumulate in FP32
}
// Reduce in FP32, then convert result back to BF16
```

## Warp-level optimization

Warp shuffles are the fastest way to reduce within a warp (no shared memory
needed):

```cuda
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}
```

For block-level reductions, combine warp shuffles with minimal shared memory:

```cuda
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}
```

This uses only `ceil(threads/32)` floats of shared memory — negligible
compared to the 100 KB available.

## Compilation

### Build flags

```bash
# For GB10 specifically
nvcc -gencode=arch=compute_120,code=sm_121 -O3 --use_fast_math kernel.cu

# In build.toml
[kernel.my_kernel]
backend = "cuda"
cuda-capabilities = ["12.1"]
src = ["kernel_src/my_kernel.cu"]
```

The sm_121 warning from PyTorch is safe to ignore:
```
Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

sm_120 and sm_121 are binary compatible. Kernels compiled for compute_120
run correctly on sm_121.

### Environment variable

Set `TORCH_CUDA_ARCH_LIST` when building with `setup.py` or `pip install`:

```bash
export TORCH_CUDA_ARCH_LIST="12.1a"
pip install -e .
```

### Flash Attention

Do **not** install `flash-attn` on GB10 — it causes `libcudart.so.12`
linking errors. PyTorch's native SDPA (Scaled Dot-Product Attention) uses
the same algorithm and is faster on Blackwell because it can target sm_121
natively.

## Profiling

```bash
# System-wide timeline
nsys profile -o gb10_profile python your_script.py

# Detailed kernel metrics
ncu --set full -o gb10_metrics python your_script.py

# Quick bandwidth check
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    python your_script.py
```

Key metrics to watch on GB10:
- **DRAM throughput %** — are you saturating the ~273 GB/s bandwidth?
- **Achieved occupancy** — aim for ≥80% (easy with 256-thread blocks)
- **Warp stall reasons** — memory latency stalls are expected; compute stalls indicate register spills

## Best practices for GB10

1. **Vectorize everything.** Use `__nv_bfloat162`, `__half2`, `float4` — the
   bandwidth gap with HBM GPUs makes every wasted byte hurt more.
2. **Fuse operations.** Reducing memory round-trips is the single highest
   impact optimization on GB10. A fused RMSNorm that reads once and writes
   once is 2.6× faster than PyTorch's decomposed version.
3. **Use 256-thread blocks.** This gives 100% occupancy (6 blocks × 8 warps
   = 48 warps/SM) and is the sweet spot for most kernels.
4. **Don't over-tile.** With 48 SMs and 24 MB L2, use smaller tile sizes
   than H100 guides suggest. `BLOCK_M=64` instead of 128.
5. **Skip pinned memory.** Unified memory makes `cudaHostAlloc` pointless.
   Just use regular `torch.Tensor` operations.
6. **Prefer BF16 for training, FP16 for inference.** Both are vectorized
   identically via 2-element packed types.
7. **Don't install flash-attn.** Use PyTorch native SDPA instead.
8. **Profile with ncu.** The bandwidth ceiling is lower, so you will hit it
   sooner — know exactly where your kernel sits relative to the peak.

## Example kernel: vectorized RMSNorm

A complete, tested RMSNorm kernel targeting GB10 is available at
[logos-flux/gb10-rmsnorm](https://huggingface.co/logos-flux/gb10-rmsnorm)
on the HuggingFace Kernel Hub.

```python
from kernels import get_kernel

kernel = get_kernel("logos-flux/gb10-rmsnorm")

out = torch.empty_like(x)
kernel.rmsnorm(out, x, weight, 1e-6)
```
