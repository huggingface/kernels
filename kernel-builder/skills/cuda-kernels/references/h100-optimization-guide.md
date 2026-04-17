# H100 GPU Optimization Guide for Diffusers Kernels

Deep dive into H100-specific optimizations for diffusion model CUDA kernels.

## H100 Hopper Architecture Overview

### Key Specifications

| Component | Specification | Notes |
|-----------|---------------|-------|
| Compute Capability | 9.0 (sm_90) | Target in build.toml |
| SMs | 132 | More than A100 (108) |
| CUDA Cores | 16,896 | 128 per SM |
| Tensor Cores | 528 | 4th gen, FP8 support |
| L2 Cache | 50 MB | 2.5x A100 |
| Shared Memory | 192 KB/SM | Configurable (96/144/192) |
| Registers | 64K 32-bit/SM | 255 per thread max |
| Memory Bandwidth | 3.35 TB/s | HBM3 |
| Max Threads/SM | 2048 | 64 warps |
| Max Threads/Block | 1024 | 32 warps |
| Warp Size | 32 | Unchanged |

### New Hopper Features

1. **Thread Block Clusters** - Groups of thread blocks that can cooperate
2. **Distributed Shared Memory** - Access shared memory across blocks in cluster
3. **Tensor Memory Accelerator (TMA)** - Hardware-accelerated bulk memory operations
4. **FP8 Support** - Native 8-bit floating point in tensor cores
5. **Asynchronous Execution** - More overlap between compute and memory

## Memory Hierarchy Optimization

### Global Memory Access Patterns

```cuda
// GOOD: Coalesced access (threads access consecutive addresses)
// Each thread reads 4 bytes, warp reads 128 bytes (one transaction)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = input[idx];

// BAD: Strided access (each thread in warp accesses different cache lines)
int idx = threadIdx.x * stride;  // Avoid stride > 1
float val = input[idx];
```

**Transaction sizes:**
- 32 bytes minimum
- 128 bytes optimal (full warp, FP32)
- Align to 128-byte boundaries when possible

### Vectorized Memory Access (Critical for Bandwidth)

Vectorized loads/stores dramatically improve memory bandwidth utilization:

**BFloat16 vectorization (2x elements per load):**
```cuda
// Load 2 bfloat16 elements at once (32-bit transaction)
const __nv_bfloat162* vec_input = reinterpret_cast<const __nv_bfloat162*>(row_input);

#pragma unroll 4
for (int i = tid; i < hidden_size / 2; i += stride) {
    __nv_bfloat162 v = vec_input[i];
    float v0 = __bfloat162float(v.x);
    float v1 = __bfloat162float(v.y);
    // Process v0, v1...
}

// Write back vectorized
__nv_bfloat162* vec_output = reinterpret_cast<__nv_bfloat162*>(row_output);
__nv_bfloat162 result;
result.x = __float2bfloat16(val0);
result.y = __float2bfloat16(val1);
vec_output[i] = result;
```

**FP16 vectorization:**
```cuda
const __half2* vec_input = reinterpret_cast<const __half2*>(row_input);
__half2 v = vec_input[i];
float v0 = __half2float(v.x);
float v1 = __half2float(v.y);
```

**FP32 vectorization (4x elements per load):**
```cuda
const float4* vec_input = reinterpret_cast<const float4*>(row_input);
float4 v = vec_input[i];
// v.x, v.y, v.z, v.w are 4 consecutive floats
```

**Benchmark Results (RMSNorm on H100):**

| Implementation | Time (ms) | Speedup |
|:---|:---:|:---:|
| Scalar loads | 0.065 | 1.00x |
| Vectorized (__nv_bfloat162) | 0.019 | **3.37x** |

**Bandwidth achieved:** 38% of H100's theoretical 3.35 TB/s

### L2 Cache Utilization

H100's 50MB L2 cache is significant for diffusion models:

```cuda
// For attention: Process Q blocks to maximize K,V cache reuse
// K,V tiles stay in L2 while Q block iterates

// Block size tuning for L2:
// BLOCK_SIZE_M = 128  (Q block)
// BLOCK_SIZE_N = 64   (K,V block)
// With head_dim=64, each tile = 128*64*2 = 16KB (FP16)
// Multiple tiles fit in L2 for reuse
```

### Shared Memory Configuration

H100 supports configurable shared memory per SM:
- 96 KB shared + 128 KB L1
- 144 KB shared + 80 KB L1
- 192 KB shared + 32 KB L1

For attention kernels with large tiles:
```cuda
// Request max shared memory
cudaFuncSetAttribute(
    attention_forward_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    192 * 1024  // 192 KB
);
```

### Bank Conflicts

Shared memory has 32 banks (4 bytes per bank):
```cuda
// Bank conflict example (all threads hit same bank)
__shared__ float data[1024];
float val = data[threadIdx.x * 32];  // BAD: 32-stride = same bank

// No bank conflict
float val = data[threadIdx.x];  // GOOD: consecutive access

// Bank conflict avoidance with padding
__shared__ float data[32][33];  // 33 instead of 32
float val = data[threadIdx.y][threadIdx.x];  // Different banks
```

## Warp-Level Optimizations

### Shuffle Instructions

Fastest way to share data within a warp:
```cuda
// Reduction using shuffles (no shared memory needed)
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Broadcast from lane 0
float broadcast = __shfl_sync(0xffffffff, val, 0);

// Butterfly shuffle for max
float max_val = __shfl_xor_sync(0xffffffff, val, 16);
val = max(val, max_val);
// ... repeat for 8, 4, 2, 1
```

### Warp-Level Matrix Operations

For small matrices, keep data in registers:
```cuda
// Example: 4x4 matrix per warp (1 element per thread in first 16 lanes)
// More efficient than shared memory for small sizes
```

## Register Optimization

### Register Pressure

H100 allows 255 registers per thread. Monitor usage:
```bash
nvcc --ptxas-options=-v your_kernel.cu
# Shows: "Used X registers, Y bytes smem"
```

### Register Tiling

For attention, keep partial results in registers:
```cuda
// Each thread maintains its own row_max and row_sum
float row_max = -INFINITY;
float row_sum = 0.0f;

// And output accumulator (fits in registers if head_dim is small)
float out_acc[HEAD_DIM];  // Works for head_dim <= ~64
```

## Occupancy Tuning

### Calculating Occupancy

```
Occupancy = Active Warps per SM / Max Warps per SM (64)

Limiting factors:
1. Registers: 65536 registers / (threads_per_block * regs_per_thread)
2. Shared Memory: 192KB / smem_per_block
3. Threads: 2048 / threads_per_block
```

### Block Size Selection

For H100 diffusers kernels:

| Kernel Type | Threads/Block | Warps | Reasoning |
|-------------|---------------|-------|-----------|
| Element-wise | 256 | 8 | High occupancy, simple |
| Reduction | 512-1024 | 16-32 | Need enough threads for full reduction |
| Attention | 256 | 8 | Balance shared mem and registers |

### Occupancy Calculator Usage

```python
# Use CUDA occupancy API
from numba import cuda
import numba.cuda as nb_cuda

@cuda.jit
def my_kernel(...):
    pass

# Get suggested block size
max_block_size = my_kernel.suggest_cooperative_groups_max_block_size()
occupancy = my_kernel.occupancy(max_block_size)
```

## Precision and Numerical Stability

### BF16 vs FP16

For diffusion models:
```
FP16: 1 sign + 5 exponent + 10 mantissa
- Better precision (10 bits)
- Smaller range (±65504)
- Risk of overflow in attention scores

BF16: 1 sign + 8 exponent + 7 mantissa
- Same range as FP32
- Less precision (7 bits)
- Safer for attention (no overflow)
- Preferred for training
```

### Online Softmax for Attention

Numerically stable softmax without materializing full attention matrix:
```cuda
// Traditional (bad for memory)
// scores = Q @ K^T  // [seq, seq] - huge!
// softmax(scores)
// output = scores @ V

// Online softmax (good)
float row_max = -INFINITY;
float row_sum = 0.0f;

for each K block:
    compute local_scores
    local_max = max(local_scores)

    // Update running statistics
    new_max = max(row_max, local_max)
    rescale = exp(row_max - new_max)

    row_sum = row_sum * rescale + sum(exp(local_scores - new_max))
    row_max = new_max

    // Update output accumulator with rescaling
    out_acc = out_acc * rescale + softmax_scores @ V_block
```

### Mixed Precision Pattern

Use FP32 for reductions, low precision for memory:
```cuda
// Input in FP16/BF16
float sum = 0.0f;  // Accumulate in FP32
for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = float(input[i]);  // Cast to FP32
    sum += val * val;
}
// Reduction in FP32
sum = block_reduce_sum(sum);

// Output in FP16/BF16
output[i] = scalar_t(result);  // Cast back
```

## Diffusers-Specific Optimizations

### LTX-Video Attention Pattern

LTX-Video uses 3D positional encoding for video:
```cuda
// Sequence layout: [batch, num_frames * height * width, heads, head_dim]
// Position encoding splits head_dim into temporal + spatial components

// Efficient 3D position decoding
int t_idx = seq_idx / (height * width);
int hw_idx = seq_idx % (height * width);
int h_idx = hw_idx / width;
int w_idx = hw_idx % width;

// Apply different RoPE frequencies to different head_dim ranges
// Typically: head_dim / 3 for each of (t, h, w)
```

### DiT Adaptive LayerNorm

DiT uses timestep-conditioned normalization:
```cuda
// Formula: norm(x) * weight * (1 + scale) + shift
// scale, shift come from MLP on timestep embedding

// Optimization: Fuse the MLP projection with AdaLN application
// Compute 6 values per block: (scale1, shift1, gate1, scale2, shift2, gate2)
// Apply to attention output and FFN output respectively
```

### GEGLU FFN Pattern

Common in modern transformers:
```cuda
// Input: [batch, seq, 2*hidden]
// Split into gate and value halves
// Output: gelu(gate) * value

// Memory optimization: Don't materialize intermediate
float gate = float(input[idx]);
float value = float(input[idx + hidden_size]);
float activated = gelu_tanh(gate) * value;
output[idx] = scalar_t(activated);
```

## Profiling and Debugging

### NVIDIA Nsight Systems (nsys)

System-wide profiling:
```bash
nsys profile -o profile_report python your_script.py

# Key metrics to watch:
# - Kernel duration
# - Memory transfer time
# - GPU idle time
# - Stream utilization
```

### NVIDIA Nsight Compute (ncu)

Detailed kernel analysis:
```bash
# Full metrics
ncu --set full -o metrics.ncu-rep python your_script.py

# Specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
python your_script.py

# Key metrics for diffusers kernels:
# - Achieved occupancy
# - Memory throughput
# - Compute throughput
# - Warp stall reasons
```

### Common Performance Issues

1. **Low occupancy**: Too many registers or shared memory
   - Solution: Reduce register usage, use smaller block sizes

2. **Memory bound**: Low compute/memory ratio
   - Solution: Fuse operations, increase arithmetic intensity

3. **Bank conflicts**: Shared memory access pattern issues
   - Solution: Add padding, change access pattern

4. **Warp divergence**: Conditional branches within warp
   - Solution: Restructure to process similar elements together

5. **Launch overhead**: Too many small kernels
   - Solution: Fuse kernels, use CUDA graphs

## CUDA Compilation Flags

```bash
# For H100 specifically
nvcc -arch=sm_90 -O3 your_kernel.cu

# Useful flags:
# -maxrregcount=N    Limit registers per thread
# --ptxas-options=-v Print register/smem usage
# -lineinfo          Add debug line info
# --use_fast_math    Fast but less precise math
# -Xptxas -dlcm=ca   Cache global loads in L1
```

## Best Practices Summary

1. **Memory Access**: Always coalesce, align to 128 bytes
2. **Shared Memory**: Use for data reuse, watch bank conflicts
3. **Registers**: Prefer for small, thread-private data
4. **Reductions**: Use warp shuffles, avoid atomics when possible
5. **Precision**: BF16 for training, FP16 for inference, FP32 for accumulation
6. **Block Size**: Start with 256, tune based on occupancy
7. **Profile**: Use ncu to identify bottlenecks before optimizing
8. **Fuse**: Combine operations to reduce memory traffic
9. **Type Conversions**: Always use explicit `to_float()`/`from_float()` helpers (PyTorch disables implicit FP16/BF16 conversions)

## Working Example

For complete, production-ready implementations, see `examples/ltx_video/`:

**Benchmark results on H100:**
| Kernel | Shape | Time |
|--------|-------|------|
| RMSNorm | [2, 1024, 2048] | 0.054 ms |
| GEGLU | [2, 1024, 4096] → [2, 1024, 2048] | 0.030 ms |
| RoPE 3D | [2, 480, 8, 64] | 1.670 ms |

**Build and test:**
```bash
cd examples/ltx_video
uv pip install -e .
uv run python generate_video.py --num-frames 13 --steps 20
```
