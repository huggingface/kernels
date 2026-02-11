
# H100 CUDA Kernels for Diffusers

This skill provides patterns and guidance for developing optimized CUDA kernels targeting NVIDIA H100 GPUs (compute capability 9.0) for use with the HuggingFace diffusers library.

## Quick Start

**For benchmarking kernel performance:**
```bash
# Benchmark with optimized kernels (6% end-to-end speedup)
python generate_video.py --use-optimized-kernels

# Benchmark baseline with torch.compile (34% speedup)
python generate_video.py --no-optimized-kernels --compile

# Compare configurations (note: --compile and --use-optimized-kernels are mutually exclusive)
python generate_video.py --use-optimized-kernels && \
python generate_video.py --no-optimized-kernels --compile
```

**For isolated kernel micro-benchmarks:**
```bash
python benchmark_rmsnorm.py
```

**For a minimal integration example (~150 lines):**
```bash
python scripts/ltx_kernel_injection_example.py
```

## When This Skill Applies

Use this skill when:
- **Benchmarking kernel performance** against baseline implementations
- Writing new CUDA kernels for diffusion models
- Optimizing existing kernels for H100 architecture
- Implementing custom attention, normalization, or activation layers
- Integrating kernels with diffusers pipelines (LTX-Video, Stable Diffusion, FLUX, DiT)
- Debugging kernel performance issues on H100

## Working Example

A complete working example is available at `examples/ltx_video/`. This demonstrates:
- Custom CUDA kernels (RMSNorm, RoPE 3D, GEGLU, AdaLN)
- Build system setup with setup.py, build.toml, and flake.nix
- PyTorch C++ bindings and Python API
- Benchmarking script for comparing optimized vs baseline performance

## Benchmarking Kernels

Use the benchmark script to measure kernel performance:

```bash
# Full benchmark with all options
python scripts/benchmark_example.py \
    --use-optimized-kernels \
    --compile \
    --batch-size 1 \
    --num-frames 161 \
    --height 512 \
    --width 768 \
    --steps 50 \
    --warmup-iterations 2
```

### Benchmark Script Options

| Option | Default | Description |
|--------|---------|-------------|
| `--use-optimized-kernels` | auto | Use custom H100 CUDA kernels |
| `--no-optimized-kernels` | - | Use baseline implementation |
| `--compile` | false | Enable torch.compile on transformer |
| `--batch-size` | 1 | Number of videos per prompt |
| `--num-frames` | 161 | Number of frames to generate |
| `--height` | 512 | Video height in pixels |
| `--width` | 768 | Video width in pixels |
| `--steps` | 50 | Denoising steps |
| `--warmup-iterations` | 2 | Warmup runs before benchmark |

### Example Benchmark Results

**End-to-End Video Generation (49 frames, 30 steps, H100 80GB):**

| Configuration | Time (s) | it/s | Speedup | Notes |
|:---|:---:|:---:|:---:|:---|
| Baseline (no compile) | 2.87 | 12.58 | 1.00x | Reference |
| **Optimized Kernels** | 2.70 | 13.52 | **1.06x** | 6% faster |
| Baseline + torch.compile | 2.14 | 19.05 | 1.34x | 34% faster |

**Important:** `--use-optimized-kernels` and `--compile` are currently mutually exclusive. Custom kernels require PyTorch custom op registration to work with torch.compile.

**Key metrics to capture:**
- **Device:** GPU model (e.g., NVIDIA H100 80GB HBM3)
- **Precision:** Data type used (e.g., bfloat16)
- **Resolution:** Width x Height (e.g., 768x512)
- **Frames:** Number of frames generated (e.g., 49, 161)

### RMSNorm Micro-benchmarks

The vectorized RMSNorm kernel achieves **2.67x average speedup** over PyTorch baseline:

| Shape | Custom (ms) | PyTorch (ms) | Speedup |
|:---|:---:|:---:|:---:|
| [1×1024×2048] | 0.019 | 0.065 | **3.37x** |
| [2×1024×2048] | 0.024 | 0.073 | **3.04x** |
| [4×1024×2048] | 0.036 | 0.093 | **2.58x** |
| [2×4096×3072] | 0.087 | 0.208 | **2.41x** |
| [4×4096×3072] | 0.157 | 0.392 | **2.49x** |

**Bandwidth efficiency:** 38% of H100's theoretical 3.35 TB/s

**Why end-to-end speedup is smaller:** RMSNorm accounts for ~5% of total compute in LTX-Video. The remaining time is spent in attention (Flash Attention/SDPA), linear projections, and VAE decode.

## Project Structure

```
.claude/skills/h100-diffusers-kernels/
├── scripts/
│   ├── benchmark_example.py        # End-to-end video generation benchmark
│   ├── benchmark_rmsnorm.py        # Isolated RMSNorm micro-benchmark
│   └── ltx_kernel_injection_example.py  # Minimal integration example
├── references/
│   ├── diffusers-integration.md    # Complete integration guide
│   ├── troubleshooting.md          # Common issues and solutions
│   ├── kernel-templates.md         # CUDA kernel templates (includes vectorized)
│   └── h100-optimization-guide.md  # H100 optimization deep dive
└── SKILL.md                        # This file

examples/ltx_video/                  # Complete working example
├── kernel_src/
│   └── rmsnorm.cu                  # Vectorized RMSNorm kernel (2.67x faster)
├── torch-ext/                      # PyTorch bindings
├── generate_video.py               # Full benchmark script
├── benchmark_rmsnorm.py            # Isolated kernel benchmark
└── setup.py                        # pip install -e .
```

## H100 Architecture Reference

| Spec | Value | Optimization Impact |
|------|-------|---------------------|
| SMs | 132 | Grid sizing: aim for multiples of 132 |
| Threads/SM | 2048 | Max 16 blocks of 128 threads per SM |
| Shared Memory | 192 KB/SM | Large tiles possible |
| L2 Cache | 50 MB | Reuse across blocks |
| Memory BW | 3.35 TB/s | Coalesced access critical |
| Warp Size | 32 | All reductions use warp shuffles |

## Core Kernel Patterns

### Vectorized Memory Access (Critical for Performance)

**BFloat16 vectorization using `__nv_bfloat162`:**
```cuda
// Load 2 bfloat16 elements at once (32-bit load)
const __nv_bfloat162* vec_input = reinterpret_cast<const __nv_bfloat162*>(row_input);

#pragma unroll 4
for (int i = tid; i < vec_hidden; i += stride) {
    __nv_bfloat162 v = vec_input[i];
    float v0 = __bfloat162float(v.x);
    float v1 = __bfloat162float(v.y);
    sum_sq += v0 * v0 + v1 * v1;
}
```

**FP16 vectorization using `__half2`:**
```cuda
const __half2* vec_input = reinterpret_cast<const __half2*>(row_input);
__half2 v = vec_input[i];
float v0 = __half2float(v.x);
float v1 = __half2float(v.y);
```

**FP32 vectorization using `float4`:**
```cuda
const float4* vec_input = reinterpret_cast<const float4*>(row_input);
float4 v = vec_input[i];
sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
```

### Warp Shuffle Reductions
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

### Block Sizes for Attention
- `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 64`, `BLOCK_SIZE_K = 64`
- `NUM_WARPS = 8`

### Thread Configuration

For element-wise ops (RoPE, GEGLU):
```cuda
constexpr int BLOCK_SIZE = 256;
int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
```

For reduction ops (LayerNorm, RMSNorm) with vectorization:
```cuda
// Divide by 2 for bf16/fp16 vectorized access
int threads = min(hidden_size / 2, MAX_THREADS);
threads = max(threads, WARP_SIZE);
threads = (threads + 32 - 1) / 32 * 32;  // Round to warp boundary
```

## Supported Data Types

All kernels support three precision modes:
- `__half` (FP16) - Default for inference
- `__nv_bfloat16` (BF16) - Preferred for training
- `float` (FP32) - Reference/debugging

## Building Kernels

### With Nix (Recommended)
```bash
nix run .#build-and-copy --max-jobs 2 --cores 8 -L
```

### With pip/uv
```bash
uv pip install -e .
```

### build.toml Configuration
```toml
[general]
name = "ltx_kernels"
backends = ["cuda"]

[kernel.your_kernel]
backend = "cuda"
src = ["kernel_src/your_kernel.cu"]
cuda-capabilities = ["9.0"]
```

## Diffusers Integration

> **See [diffusers-integration.md](references/diffusers-integration.md) for the complete guide.**

### Critical Pitfalls

#### 1. RMSNorm Weight May Be None

LTX-Video uses `elementwise_affine=False` for some RMSNorm modules:
```python
# Transformer blocks: NO WEIGHT
self.norm1 = RMSNorm(dim, elementwise_affine=False)

# Attention modules: HAS WEIGHT
self.norm_q = torch.nn.RMSNorm(..., elementwise_affine=True)
```

**Solution:** Handle both cases:
```python
has_weight = hasattr(module, 'weight') and module.weight is not None
if has_weight:
    output = rmsnorm(x, module.weight, eps=eps)
else:
    weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    output = rmsnorm(x, weight, eps=eps)
```

#### 2. Diffusers RMSNorm != torch.nn.RMSNorm

```python
# WRONG - misses diffusers RMSNorm
if isinstance(module, torch.nn.RMSNorm):

# CORRECT - catches all RMSNorm variants
if type(module).__name__ == 'RMSNorm':
```

#### 3. LTX-Video Uses GELU, Not GEGLU

LTX-Video uses `activation_fn="gelu-approximate"`. Don't patch GEGLU for LTX-Video.

#### 4. Inject Kernels BEFORE CPU Offloading

```python
pipe = LTXPipeline.from_pretrained(...)
pipe.to("cuda")
inject_optimized_kernels(pipe)  # BEFORE offloading
pipe.enable_model_cpu_offload()  # Now safe
```

### Minimal Integration Pattern

```python
from diffusers import LTXPipeline
from ltx_kernels import rmsnorm

def patch_rmsnorm_modules(model):
    """Patch all RMSNorm modules to use custom kernel."""
    for name, module in model.named_modules():
        if type(module).__name__ == 'RMSNorm':
            eps = getattr(module, 'eps', 1e-6)
            has_weight = hasattr(module, 'weight') and module.weight is not None

            if has_weight:
                def make_forward(mod, epsilon):
                    def forward(x):
                        return rmsnorm(x, mod.weight, eps=epsilon)
                    return forward
                module.forward = make_forward(module, eps)
            else:
                def make_forward(epsilon):
                    def forward(x):
                        w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
                        return rmsnorm(x, w, eps=epsilon)
                    return forward
                module.forward = make_forward(eps)

# Usage
pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")
patch_rmsnorm_modules(pipe.transformer)
pipe.enable_model_cpu_offload()
```

## Kernel-Specific Guidelines

### RMSNorm
- Input layout: `[..., hidden_size]`
- Epsilon default: 1e-6
- **Weight may be None** if `elementwise_affine=False`
- **Vectorization:** Use `__nv_bfloat162` for BF16, `__half2` for FP16, `float4` for FP32
- **Performance:** 2.67x faster than PyTorch with vectorized implementation
- **Bandwidth:** Achieves ~38% of H100's 3.35 TB/s theoretical bandwidth

### RoPE
- 1D: `[batch, seq, heads, head_dim]` - for text
- 3D: `[batch, t*h*w, heads, head_dim]` - for video
- LTX-Video computes its own RoPE via `LTXVideoRotaryPosEmbed`

### GEGLU vs GELU
- **GEGLU**: Input `[batch, seq, 2*hidden]` -> Output `[batch, seq, hidden]`
- **GELU**: Standard activation
- **LTX-Video uses GELU, NOT GEGLU**

### AdaLN
- Formula: `norm(x) * weight * (1 + scale) + shift`
- Used in DiT blocks for conditioning

## Performance Profiling

```bash
# NVIDIA Nsight Systems
nsys profile -o profile python your_script.py

# NVIDIA Nsight Compute
ncu --set full -o metrics python your_script.py
```

## Common Issues

> **See [troubleshooting.md](references/troubleshooting.md) for all common issues and solutions.**

Quick fixes:
- **"NoneType has no attribute contiguous"**: RMSNorm weight is None, create ones
- **isinstance() not matching**: Use `type(module).__name__` instead
- **GEGLU not called**: Model uses GELU, not GEGLU
- **Patching doesn't persist**: Inject before `enable_model_cpu_offload()`
- **torch.compile fails with custom kernels**: See below

### torch.compile Compatibility

Custom CUDA kernels and `torch.compile` are **mutually exclusive** unless you register the kernel as a PyTorch custom op.

**Error message:**
```
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
```

**Workaround options:**
1. Use `--use-optimized-kernels` without `--compile` (6% speedup)
2. Use `--compile` without custom kernels (34% speedup)
3. Register kernel as custom op (advanced, requires `torch.library`)

**To register as custom op (for torch.compile compatibility):**
```python
import torch

@torch.library.custom_op("ltx_kernels::rmsnorm", mutates_args={"out"})
def rmsnorm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    ops.rmsnorm_forward(out, input.contiguous(), weight.contiguous(), eps)

@rmsnorm.register_fake
def _(out, input, weight, eps):
    pass  # No shape changes
```

## See Also

- [benchmark_example.py](scripts/benchmark_example.py) - **Benchmarking script for comparing optimized vs baseline - START HERE**
- [ltx_kernel_injection_example.py](scripts/ltx_kernel_injection_example.py) - Minimal integration example (~150 lines)
- [diffusers-integration.md](references/diffusers-integration.md) - Complete integration guide
- [troubleshooting.md](references/troubleshooting.md) - Common issues and solutions
- [kernel-templates.md](references/kernel-templates.md) - Complete kernel templates
- [h100-optimization-guide.md](references/h100-optimization-guide.md) - Deep dive on H100 optimizations
- [examples/ltx_video/](../../../examples/ltx_video/) - Full LTX-Video example directory
