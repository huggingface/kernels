# T4 GPU Optimization Guide for Diffusers/Transformers Kernels

Deep dive into T4-specific optimizations for diffusion model and LLM CUDA kernels. The T4 is a Turing architecture GPU commonly found in cloud instances (GCP, AWS, etc.) and is popular for inference workloads.

## T4 Turing Architecture Overview

### Key Specifications

| Component | T4 | Notes |
|-----------|-----|-------|
| Compute Capability | 7.5 (sm_75) | Target in build.toml |
| SMs | 40 | Much fewer than A100 (108) or H100 (132) |
| CUDA Cores | 2,560 | 64 per SM |
| Tensor Cores | 320 | 2nd gen, FP16/INT8 |
| L2 Cache | 4 MB | Much smaller than A100/H100 |
| Shared Memory | 64 KB/SM | Configurable (32/48/64) |
| Registers | 64K 32-bit/SM | 255 per thread max |
| Memory Bandwidth | 320 GB/s | GDDR6 (not HBM!) |
| Memory | 16 GB | GDDR6 |
| Max Threads/SM | 1024 | 32 warps (half of A100!) |
| Max Threads/Block | 1024 | 32 warps |
| Warp Size | 32 | Unchanged |
| TDP | 70W | Very power efficient |

### T4 vs A100 vs H100 Comparison

| Feature | T4 | A100 | H100 | Notes |
|---------|-----|------|------|-------|
| Memory BW | 320 GB/s | 2.0 TB/s | 3.35 TB/s | T4 is 6x slower |
| SMs | 40 | 108 | 132 | T4 has ~30% of SMs |
| Shared Mem/SM | 64 KB | 164 KB | 192 KB | T4 needs smaller tiles |
| L2 Cache | 4 MB | 40 MB | 50 MB | T4 limited cache reuse |
| Memory | 16 GB | 40-80 GB | 80 GB | T4 requires careful memory |
| BF16 Support | **No** | Yes | Yes | T4 only FP16! |
| Max Threads/SM | 1024 | 2048 | 2048 | T4 half occupancy |

### Key T4 Constraints

1. **No BFloat16** - Must use FP16 for half-precision
2. **Limited Memory** - 16GB requires careful batching
3. **Lower Bandwidth** - 320 GB/s limits memory-bound kernels
4. **Fewer SMs** - Less parallelism, smaller grid sizes
5. **Smaller Shared Memory** - 64 KB/SM limits tile sizes
6. **Half Max Threads** - 1024/SM instead of 2048

## Memory Considerations

### FP16 Instead of BF16

**Critical:** T4 does not support BF16. Use FP16:

```python
# T4: Use FP16
model = model.to(torch.float16)

# A100/H100: Can use BF16
# model = model.to(torch.bfloat16)
```

```cuda
// T4: Use __half, NOT __nv_bfloat16
const __half2* vec_input = reinterpret_cast<const __half2*>(row_input);
__half2 v = vec_input[i];
float v0 = __half2float(v.x);
float v1 = __half2float(v.y);

// Write back
__half2 result;
result.x = __float2half(val0);
result.y = __float2half(val1);
vec_output[i] = result;
```

### Memory-Bound Kernel Optimization

T4's 320 GB/s bandwidth is the main bottleneck. Maximize arithmetic intensity:

```cuda
// BAD: Low arithmetic intensity (memory bound on T4)
for (int i = tid; i < size; i += stride) {
    output[i] = input[i] * scale;  // 1 multiply per 2 loads
}

// BETTER: Fuse operations to increase arithmetic intensity
for (int i = tid; i < size; i += stride) {
    float val = input[i];
    val = val * scale + bias;
    val = max(val, 0.0f);  // ReLU
    output[i] = val;  // More ops per memory access
}
```

### Vectorized Memory Access

Even more critical on T4 due to lower bandwidth:

**FP16 vectorization (2x elements per load):**
```cuda
const __half2* vec_input = reinterpret_cast<const __half2*>(row_input);

#pragma unroll 4
for (int i = tid; i < hidden_size / 2; i += stride) {
    __half2 v = vec_input[i];
    float v0 = __half2float(v.x);
    float v1 = __half2float(v.y);
    sum_sq += v0 * v0 + v1 * v1;
}
```

**FP32 vectorization (4x elements per load):**
```cuda
const float4* vec_input = reinterpret_cast<const float4*>(row_input);
float4 v = vec_input[i];
// v.x, v.y, v.z, v.w are 4 consecutive floats
```

### Expected T4 Performance

| Kernel | T4 (ms) | A100 (ms) | H100 (ms) | T4 vs H100 |
|--------|---------|-----------|-----------|------------|
| RMSNorm [2, 1024, 2048] | ~0.5 | ~0.08 | 0.054 | 9x slower |
| GEGLU [2, 1024, 4096] | ~0.3 | ~0.05 | 0.030 | 10x slower |

**Bandwidth achieved:** Target 40-50% of T4's 320 GB/s theoretical

## Shared Memory Configuration

T4 supports configurable shared memory per SM:
- 32 KB shared + 32 KB L1
- 48 KB shared + 16 KB L1
- 64 KB shared + 0 KB L1 (max)

For T4, use smaller tile sizes:

```cuda
// Request shared memory (max 64 KB on T4)
cudaFuncSetAttribute(
    kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    64 * 1024  // 64 KB max on T4
);
```

### Tile Size Adjustments

Reduce tile sizes compared to A100/H100:

```cuda
// H100/A100 attention tile sizes
// BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 64

// T4 attention tile sizes (smaller due to shared memory limits)
constexpr int BLOCK_SIZE_M = 64;   // Reduced
constexpr int BLOCK_SIZE_N = 32;   // Reduced
constexpr int BLOCK_SIZE_K = 32;   // Reduced
```

## Occupancy Tuning

### Block Size Selection for T4

Due to max 1024 threads/SM (vs 2048 on A100/H100):

| Kernel Type | Threads/Block | Warps | Reasoning |
|-------------|---------------|-------|-----------|
| Element-wise | 256 | 8 | Balance occupancy |
| Reduction | 256-512 | 8-16 | Avoid over-subscription |
| Attention | 128-256 | 4-8 | Small tiles |

### Grid Sizing

For T4 with 40 SMs:

```cuda
// Aim for multiples of 40 blocks
int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
// Round up to multiple of 40 for full SM utilization
num_blocks = ((num_blocks + 39) / 40) * 40;
```

## Numerical Stability with FP16

FP16 has smaller range than BF16, requiring more care:

```
FP16: 1 sign + 5 exponent + 10 mantissa
- Range: ±65504
- Risk of overflow in attention scores!

BF16: 1 sign + 8 exponent + 7 mantissa (NOT AVAILABLE ON T4)
- Range: Same as FP32
```

### Attention Score Scaling

```cuda
// Scale attention scores to prevent FP16 overflow
float scale = 1.0f / sqrtf((float)head_dim);
// For T4 FP16: May need additional scaling
// scale *= 0.125f;  // Extra scaling if overflow occurs
```

### Mixed Precision Pattern

Always accumulate in FP32:

```cuda
// Input in FP16 (T4)
float sum = 0.0f;  // Accumulate in FP32
for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = __half2float(input[i]);  // Convert to FP32
    sum += val * val;
}
// Reduction in FP32
sum = block_reduce_sum(sum);

// Output in FP16
output[i] = __float2half(result);
```

## Build Configuration

### build.toml for T4

```toml
[general]
name = "ltx_kernels"
backends = ["cuda"]

[kernel.your_kernel]
backend = "cuda"
src = ["kernel_src/your_kernel.cu"]
cuda-capabilities = ["7.5"]  # sm_75 for T4
```

### Multi-GPU Support (T4 + A100 + H100)

```toml
[kernel.your_kernel]
backend = "cuda"
src = ["kernel_src/your_kernel.cu"]
cuda-capabilities = ["7.5", "8.0", "9.0"]  # T4, A100, H100
```

### CUDA Compilation Flags

```bash
# For T4 specifically
nvcc -arch=sm_75 -O3 your_kernel.cu

# For T4 + A100 + H100
nvcc -gencode=arch=compute_75,code=sm_75 \
     -gencode=arch=compute_80,code=sm_80 \
     -gencode=arch=compute_90,code=sm_90 \
     -O3 your_kernel.cu
```

## T4-Specific Optimizations

### INT8 Quantization

T4 tensor cores support INT8 for fast inference:

```python
# PyTorch dynamic quantization
from torch.quantization import quantize_dynamic
model_int8 = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

### TensorRT Optimization

T4 is commonly used with TensorRT:

```python
import torch_tensorrt

# Compile model for T4
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224], dtype=torch.float16)],
    enabled_precisions={torch.float16}
)
```

### Batch Size Considerations

With only 16GB memory:

```python
# Calculate max batch size
# Model: ~3GB (FP16)
# Activations: ~2GB per batch
# Max batch size: (16 - 3) / 2 ≈ 6

# Use gradient checkpointing for training
model.gradient_checkpointing_enable()
```

## Memory Management

### 16GB Memory Strategies

1. **Enable CPU Offload**
```python
pipe.enable_model_cpu_offload()
```

2. **Use Sequential Processing**
```python
pipe.enable_sequential_cpu_offload()
```

3. **Reduce Resolution/Frames**
```python
# Lower resolution for T4
output = pipe(
    prompt="...",
    height=256,   # Reduced from 512
    width=384,    # Reduced from 768
    num_frames=9  # Reduced from 49
)
```

4. **FP16 Everywhere**
```python
pipe = pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
```

## Performance Profiling

### Nsight Profiling

```bash
nsys profile -o t4_profile python your_script.py
ncu --set full -o t4_metrics.ncu-rep python your_script.py

# Key T4 metrics:
# - Memory throughput (target 40-50% of 320 GB/s)
# - SM utilization (target high with 40 SMs)
# - Occupancy (max 1024 threads/SM)
```

### Common T4 Bottlenecks

1. **Memory Bandwidth** - 320 GB/s is the main limit
2. **Limited Memory** - 16GB requires careful management
3. **No BF16** - Must handle FP16 overflow risks
4. **Smaller Tiles** - 64KB shared memory limits

## Migration from H100/A100 to T4

### Required Changes

1. **Precision**: BF16 → FP16
2. **Shared Memory**: Reduce tile sizes (192→64 KB)
3. **Grid Size**: Adjust for 40 SMs
4. **Occupancy**: Account for 1024 max threads/SM
5. **Memory**: Handle 16GB limit

### Conditional Compilation

```cuda
#if __CUDA_ARCH__ >= 800
    // A100/H100: Use BF16
    typedef __nv_bfloat16 half_t;
    typedef __nv_bfloat162 half2_t;
#else
    // T4/Turing: Use FP16
    typedef __half half_t;
    typedef __half2 half2_t;
#endif
```

### Runtime Detection

```python
import torch

def get_optimal_config():
    capability = torch.cuda.get_device_capability()

    if capability >= (9, 0):  # H100
        return {"dtype": torch.bfloat16, "batch_size": 8}
    elif capability >= (8, 0):  # A100
        return {"dtype": torch.bfloat16, "batch_size": 4}
    else:  # T4 and older
        return {"dtype": torch.float16, "batch_size": 1}
```

## Best Practices Summary (T4)

1. **Use FP16**: BF16 not supported, handle overflow carefully
2. **Vectorization**: Critical due to low bandwidth
3. **Smaller Tiles**: 64 KB shared memory limit
4. **Grid Size**: Multiples of 40 for full utilization
5. **Block Size**: 256 threads is good default
6. **Occupancy**: Max 1024 threads/SM
7. **Memory**: Plan for 16GB limit
8. **INT8**: Consider quantization for inference
9. **Profile**: Focus on memory throughput

## Working Example

```bash
cd examples/ltx_video

# Build for T4
# Ensure build.toml includes cuda-capabilities = ["7.5"]
uv pip install -e .

# Run with T4-appropriate settings
python generate_video.py \
    --use-optimized-kernels \
    --height 256 \
    --width 384 \
    --num-frames 9
```

## T4 Cloud Instance Notes

| Provider | Instance Type | Notes |
|----------|---------------|-------|
| GCP | n1-standard-4 + T4 | Most common |
| AWS | g4dn.xlarge | 1x T4 |
| AWS | g4dn.12xlarge | 4x T4 |
| Azure | NC4as T4 v3 | 1x T4 |

T4 is optimized for inference, not training. Consider A100/H100 for training workloads.
