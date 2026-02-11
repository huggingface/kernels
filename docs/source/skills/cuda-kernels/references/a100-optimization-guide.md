# A100 GPU Optimization Guide for Diffusers/Transformers Kernels

Deep dive into A100-specific optimizations for diffusion model and LLM CUDA kernels.

## A100 Ampere Architecture Overview

### Key Specifications

| Component | A100 40GB | A100 80GB | Notes |
|-----------|-----------|-----------|-------|
| Compute Capability | 8.0 (sm_80) | 8.0 (sm_80) | Target in build.toml |
| SMs | 108 | 108 | Fewer than H100 (132) |
| CUDA Cores | 6,912 | 6,912 | 64 per SM |
| Tensor Cores | 432 | 432 | 3rd gen, TF32 support |
| L2 Cache | 40 MB | 40 MB | Less than H100 (50 MB) |
| Shared Memory | 164 KB/SM | 164 KB/SM | Configurable |
| Registers | 64K 32-bit/SM | 64K 32-bit/SM | 255 per thread max |
| Memory Bandwidth | 1.55 TB/s | 2.0 TB/s | HBM2e |
| Max Threads/SM | 2048 | 2048 | 64 warps |
| Max Threads/Block | 1024 | 1024 | 32 warps |
| Warp Size | 32 | 32 | Unchanged |

### A100 vs H100 Comparison

| Feature | A100 | H100 | Impact |
|---------|------|------|--------|
| Memory BW | 2.0 TB/s | 3.35 TB/s | H100 67% faster for memory-bound |
| SMs | 108 | 132 | H100 22% more parallelism |
| Shared Mem/SM | 164 KB | 192 KB | H100 allows larger tiles |
| L2 Cache | 40 MB | 50 MB | H100 better cache utilization |
| Tensor Cores | 3rd gen | 4th gen | H100 has FP8, better throughput |
| TMA | No | Yes | H100 has hardware memory accelerator |

### Key A100 Features

1. **Third-Gen Tensor Cores** - FP16, BF16, TF32, INT8, INT4
2. **Multi-Instance GPU (MIG)** - Partition into up to 7 instances
3. **Structural Sparsity** - 2:4 sparsity support in tensor cores
4. **TF32 Mode** - FP32-like range with FP16-like throughput
5. **Asynchronous Copy** - Overlap compute and memory

## Memory Hierarchy Optimization

### Global Memory Access Patterns

Same principles as H100, but lower bandwidth makes coalescing even more critical:

```cuda
// GOOD: Coalesced access
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = input[idx];

// BAD: Strided access (even worse on A100 due to lower bandwidth)
int idx = threadIdx.x * stride;
float val = input[idx];
```

**A100 Transaction sizes:**
- 32 bytes minimum
- 128 bytes optimal (full warp, FP32)
- Memory-bound kernels more limited by 2.0 TB/s (vs 3.35 TB/s on H100)

### Vectorized Memory Access

Same vectorization patterns work on A100:

**BFloat16 vectorization:**
```cuda
const __nv_bfloat162* vec_input = reinterpret_cast<const __nv_bfloat162*>(row_input);

#pragma unroll 4
for (int i = tid; i < hidden_size / 2; i += stride) {
    __nv_bfloat162 v = vec_input[i];
    float v0 = __bfloat162float(v.x);
    float v1 = __bfloat162float(v.y);
}
```

**Expected A100 Performance (RMSNorm):**

| Implementation | A100 Time (ms) | H100 Time (ms) | A100 Speedup |
|:---|:---:|:---:|:---:|
| Scalar loads | ~0.10 | 0.065 | 1.00x |
| Vectorized | ~0.03 | 0.019 | ~3x |

**Bandwidth achieved:** Target 30-40% of A100's 2.0 TB/s theoretical

### L2 Cache Utilization

A100's 40MB L2 cache is still significant:

```cuda
// For attention: Same block size tuning works
// BLOCK_SIZE_M = 128  (Q block)
// BLOCK_SIZE_N = 64   (K,V block)
// Tiles fit in L2 for reuse
```

### Shared Memory Configuration

A100 supports configurable shared memory per SM:
- 48 KB shared + 80 KB L1 (default)
- 96 KB shared + 32 KB L1
- 164 KB shared + 0 KB L1 (max)

For attention kernels:
```cuda
// Request max shared memory
cudaFuncSetAttribute(
    attention_forward_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    164 * 1024  // 164 KB max on A100
);
```

## Warp-Level Optimizations

### Shuffle Instructions

Same warp shuffle patterns work on A100:

```cuda
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}
```

## Occupancy Tuning

### Block Size Selection for A100

| Kernel Type | Threads/Block | Warps | Reasoning |
|-------------|---------------|-------|-----------|
| Element-wise | 256 | 8 | High occupancy |
| Reduction | 512-1024 | 16-32 | Full reduction |
| Attention | 256 | 8 | Balance shared mem |

### Grid Sizing

For A100 with 108 SMs:

```cuda
// Aim for multiples of 108 blocks
int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
// Round up to multiple of 108 for full SM utilization
num_blocks = ((num_blocks + 107) / 108) * 108;
```

## Precision and Tensor Cores

### TF32 Mode (A100 Specific)

TF32 provides FP32-like range with better throughput:

```python
# Enable TF32 for matmuls (PyTorch)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

```cuda
// CUDA: Use TF32 tensor core operations
// Automatically enabled for FP32 inputs on Ampere+ with cuBLAS/cuDNN
```

### BF16 vs FP16 on A100

```
FP16: Good precision, risk of overflow
BF16: Same range as FP32, preferred for training
TF32: Best throughput for FP32-like accuracy (A100 specific)
```

## Build Configuration

### build.toml for A100

```toml
[general]
name = "ltx_kernels"
backends = ["cuda"]

[kernel.your_kernel]
backend = "cuda"
src = ["kernel_src/your_kernel.cu"]
cuda-capabilities = ["8.0"]  # sm_80 for A100
```

### Multi-GPU Support (A100 + H100)

```toml
[kernel.your_kernel]
backend = "cuda"
src = ["kernel_src/your_kernel.cu"]
cuda-capabilities = ["8.0", "9.0"]  # Both A100 and H100
```

### CUDA Compilation Flags

```bash
# For A100 specifically
nvcc -arch=sm_80 -O3 your_kernel.cu

# For both A100 and H100
nvcc -gencode=arch=compute_80,code=sm_80 \
     -gencode=arch=compute_90,code=sm_90 \
     -O3 your_kernel.cu
```

## A100-Specific Optimizations

### Async Memory Copy

A100 introduced async memory copy (cp.async):

```cuda
// Async copy from global to shared memory
__pipeline_memcpy_async(shared_ptr, global_ptr, size);
__pipeline_commit();
__pipeline_wait_prior(0);
```

### Structural Sparsity

A100 tensor cores support 2:4 sparsity (50% zeros):

```python
# PyTorch sparse semi-structured
from torch.sparse import to_sparse_semi_structured
sparse_weight = to_sparse_semi_structured(dense_weight)
```

### Multi-Instance GPU (MIG)

For inference workloads:
```bash
# Create MIG instances
nvidia-smi mig -cgi 9,9,9,9,9,9,9 -C
# Creates 7x 5GB instances on A100 40GB
```

## Performance Profiling

### Expected Performance (A100 vs H100)

| Kernel | A100 (ms) | H100 (ms) | H100 Speedup |
|--------|-----------|-----------|--------------|
| RMSNorm [2, 1024, 2048] | ~0.08 | 0.054 | 1.5x |
| GEGLU [2, 1024, 4096] | ~0.05 | 0.030 | 1.7x |

### Nsight Profiling

```bash
# Same commands work on A100
nsys profile -o a100_profile python your_script.py
ncu --set full -o a100_metrics.ncu-rep python your_script.py

# Key A100 metrics to watch:
# - sm__throughput.avg.pct_of_peak_sustained_elapsed
# - dram__throughput.avg.pct_of_peak_sustained_elapsed
# - gpu__time_duration.avg (kernel time)
```

## Migration from H100 to A100

### Code Changes Required

1. **Shared Memory**: Reduce max shared memory from 192KB to 164KB
2. **Grid Size**: Adjust for 108 SMs instead of 132
3. **No TMA**: Can't use Tensor Memory Accelerator
4. **No FP8**: Must use FP16/BF16 instead

### Backward Compatible Pattern

```cuda
// Works on both A100 and H100
#if __CUDA_ARCH__ >= 900
    // H100-specific optimizations (TMA, etc.)
#else
    // A100/older GPU fallback
#endif
```

## Best Practices Summary (A100)

1. **Memory Access**: Even more critical due to lower bandwidth
2. **Vectorization**: Use `__nv_bfloat162`, `__half2`, `float4`
3. **TF32**: Enable for FP32 workloads for ~8x speedup
4. **Block Size**: 256 threads is good default
5. **Shared Memory**: Max 164 KB/SM
6. **Grid Size**: Multiples of 108 for full utilization
7. **Profile**: Compare achieved vs theoretical bandwidth
8. **Multi-arch**: Build for both sm_80 and sm_90

## Working Example

```bash
cd examples/ltx_video

# Build for A100
# Ensure build.toml includes cuda-capabilities = ["8.0"]
uv pip install -e .

# Run benchmark
python generate_video.py --use-optimized-kernels
```
