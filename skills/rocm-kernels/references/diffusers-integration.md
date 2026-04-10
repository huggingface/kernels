# Diffusers Pipeline Integration Guide (ROCm)

Integrating custom Triton kernels into HuggingFace diffusers pipelines on AMD GPUs.

## Dependencies

Use the existing dependency file:

```bash
python -m pip install -r skills/rocm-kernels/scripts/requirements.txt
```

### Required (for this guide)

- `torch`
- `triton`
- `diffusers`
- `transformers`
- `accelerate`

### Optional

- Video export: `imageio-ffmpeg` (pip) or system `ffmpeg`
- Profiling/trace: `rocprof`, `rocprofv3`

## Tested Environment

| Component | Version |
|-----------|---------|
| GPU | AMD Radeon Graphics (gfx1201) |
| ROCm | 7.1.52802-26aae437f6 |
| PyTorch | 2.8.0+rocm7.1.1 |
| Triton | 3.4.0+rocm7.1.1 |
| diffusers | 0.37.0 |
| transformers | 4.57.3 |
| accelerate | 1.12.0 |

## Clean Install + Smoke Test

```bash
python -m venv .venv-rocm-kernels
source .venv-rocm-kernels/bin/activate
python -m pip install --upgrade pip
python -m pip install -r skills/rocm-kernels/scripts/requirements.txt
```

```bash
python - <<'PY'
import torch
from diffusers import LTXPipeline

assert torch.cuda.is_available(), "ROCm device is not available"
pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")
_ = pipe(
    prompt="A quick smoke test on ROCm",
    num_frames=9,
    height=480,
    width=704,
    num_inference_steps=1,
    guidance_scale=7.5,
)
print("Smoke test passed.")
PY
```

## Overview

This guide covers injecting optimized Triton kernels (RMSNorm, RoPE 3D, GEGLU, AdaLN) into diffusers pipelines running on ROCm. The patterns are analogous to the CUDA kernel integration but use Triton instead of CUDA C.

## LTX-Video Architecture

### Module Inventory

```python
from diffusers import LTXPipeline
pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)

# Analyze RMSNorm modules
for name, module in pipe.transformer.named_modules():
    if 'Norm' in type(module).__name__:
        has_weight = hasattr(module, 'weight') and module.weight is not None
        print(f"{name}: {type(module).__name__} (has_weight={has_weight})")
```

### Kernel Applicability in LTX-Video

| Kernel | Used? | Count | Notes |
|--------|-------|-------|-------|
| **RMSNorm** | Yes | **168** | 56 with weights, 112 without |
| **RoPE 3D** | Indirect | 1 | Diffusers computes via LTXVideoRotaryPosEmbed |
| **GEGLU** | **No** | 0 | LTX uses `activation_fn="gelu-approximate"` |
| **AdaLN** | Partial | ~28 | Scale/shift pattern in transformer blocks |

## Integration Pattern

### Step 1: Triton RMSNorm Wrapper

```python
import os
os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr,
    stride_x_row, D,
    eps: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(x_ptr + row * stride_x_row + offs, mask=mask, other=0.0).to(tl.float32)
    sq_sum = tl.sum(x * x, axis=0)
    rms_inv = tl.rsqrt(sq_sum / D + eps)

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        out = x * rms_inv * w
    else:
        out = x * rms_inv

    tl.store(out_ptr + row * stride_x_row + offs, out.to(x.dtype), mask=mask)


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6):
    """Drop-in replacement for RMSNorm forward pass."""
    x_flat = x.contiguous().view(-1, x.shape[-1])
    out = torch.empty_like(x_flat)
    M, D = x_flat.shape
    has_weight = weight is not None

    if not has_weight:
        weight = torch.ones(D, device=x.device, dtype=x.dtype)

    # CRITICAL: BLOCK_D must be >= D. Never autotune BLOCK_D.
    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    _rmsnorm_kernel[(M,)](
        x_flat, weight, out,
        x_flat.stride(0), D,
        eps, has_weight,
        BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=2,
    )
    return out.view_as(x)
```

### Step 2: Module Patcher

```python
def patch_rmsnorm_modules(model) -> int:
    """
    Patch all RMSNorm modules to use Triton kernel on ROCm.

    Handles both:
    - Modules WITH weight (elementwise_affine=True) — attention norm_q/norm_k
    - Modules WITHOUT weight (elementwise_affine=False) — transformer block norms
    """
    patched = 0
    for name, module in model.named_modules():
        # IMPORTANT: Use class name, NOT isinstance
        if type(module).__name__ == 'RMSNorm':
            eps = getattr(module, 'eps', 1e-6)
            has_weight = hasattr(module, 'weight') and module.weight is not None

            if has_weight:
                def make_forward(mod, epsilon):
                    def forward(x):
                        return triton_rmsnorm(x, mod.weight, eps=epsilon)
                    return forward
                module.forward = make_forward(module, eps)
            else:
                def make_forward_no_weight(epsilon):
                    def forward(x):
                        return triton_rmsnorm(x, None, eps=epsilon)
                    return forward
                module.forward = make_forward_no_weight(eps)

            patched += 1
    return patched
```

### Step 3: Pipeline Injection

```python
def inject_optimized_kernels(pipe) -> dict:
    """
    Inject Triton kernels into LTX-Video pipeline.

    Call AFTER pipe.to("cuda"), BEFORE pipe.enable_model_cpu_offload().
    """
    stats = {'rmsnorm_modules': 0}

    if not hasattr(pipe, 'transformer'):
        print("WARNING: Pipeline has no 'transformer' attribute!")
        return stats

    stats['rmsnorm_modules'] = patch_rmsnorm_modules(pipe.transformer)
    return stats
```

### Step 4: Usage

```python
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")  # ROCm via HIP

stats = inject_optimized_kernels(pipe)
print(f"RMSNorm modules patched: {stats['rmsnorm_modules']}")
# Expected: 168

pipe.enable_model_cpu_offload()  # AFTER injection

output = pipe(
    prompt="A cat sleeping in the sun",
    num_frames=25, height=480, width=704,
    num_inference_steps=30,
)
export_to_video(output.frames[0], "output.mp4", fps=24)
```

## Model-Specific Notes

### LTX-Video
- Uses **GELU** (`activation_fn="gelu-approximate"`), NOT GEGLU
- RMSNorm in blocks: `elementwise_affine=False` (no weight)
- RMSNorm in attention: `elementwise_affine=True` (has weight)
- RoPE: Computed by diffusers via `LTXVideoRotaryPosEmbed`

### SD3 / FLUX
- Uses **GEGLU** in FeedForward blocks
- Different attention patterns
- May have different normalization conventions
- Verify architecture before applying LTX-Video patterns

## ROCm-Specific Considerations

### BF16 vs FP16

```python
# MI355X supports BF16 — use it for diffusers
pipe = LTXPipeline.from_pretrained(..., torch_dtype=torch.bfloat16)

# R9700 (RDNA4) — check BF16 support, may need FP16
# torch_dtype=torch.float16
```

### ROCm Memory Management

```python
# ROCm uses same API as CUDA via HIP
pipe.to("cuda")                    # Works on ROCm
pipe.enable_model_cpu_offload()    # Works on ROCm
torch.cuda.empty_cache()           # Works on ROCm
```

### Triton on ROCm vs CUDA C Kernels

| Aspect | CUDA C (original skill) | Triton (this skill) |
|--------|------------------------|---------------------|
| Build system | setup.py + nvcc | No build needed |
| Portability | NVIDIA only | AMD + NVIDIA |
| Performance | Maximum | 80-95% of CUDA C |
| Complexity | High (C++/CUDA) | Lower (Python) |
| Autotune | Manual | `@triton.autotune` |
| torch.compile | Needs custom op | Automatic compatibility |

## Verification

```python
# Check injection worked
for name, module in pipe.transformer.named_modules():
    if type(module).__name__ == 'RMSNorm':
        x = torch.randn(1, 10, 2048, device='cuda', dtype=torch.bfloat16)
        out = module(x)
        print(f"RMSNorm forward: {x.shape} -> {out.shape}")
        break

# Compare with PyTorch reference
def pytorch_rmsnorm(x, weight, eps=1e-6):
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return x * rms * weight
    return x * rms

# Verify correctness
torch.testing.assert_close(
    triton_rmsnorm(x, weight, eps=1e-6),
    pytorch_rmsnorm(x, weight, eps=1e-6),
    rtol=1e-2, atol=1e-3
)
```

### Benchmark Evidence (R9700)

For reviewer-facing benchmark materials, see `examples/ltx-video-benchmark/`.

| Mode | gen_time_s | time_per_step_s | peak_memory_gb | speedup |
|------|-----------:|----------------:|---------------:|--------:|
| baseline (mean of 3) | 6.91 | 0.231 | 18.58 | 1.00x |
| triton (mean of 3) | 6.10 | 0.203 | 18.58 | 1.13x |
| compile (single run) | 5.05 | 0.168 | 18.58 | 1.37x |

Videos are generated under `examples/ltx-video-benchmark/`:

- `benchmark_results.json`
- `ltx_video_baseline.mp4`
- `ltx_video_triton.mp4`
- `ltx_video_compile.mp4`
- `trace/codex_trace.json`
- `trace/opencode_trace.jsonl`
- `trace/opencode_trace_result.json`

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `NoneType has no attribute contiguous` | RMSNorm weight is None, pass `None` to kernel |
| `isinstance()` not matching | Use `type(module).__name__ == 'RMSNorm'` |
| GEGLU not called | LTX-Video uses GELU, not GEGLU |
| Patching doesn't persist | Inject BEFORE `enable_model_cpu_offload()` |
| HIP error during inference | Check ROCm version compatibility with PyTorch |
