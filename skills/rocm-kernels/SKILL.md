---
name: rocm-kernels
description: "Provides guidance for writing and benchmarking optimized Triton kernels for AMD GPUs (MI355X, R9700) on ROCm, targeting HuggingFace diffusers (LTX-Video, SD3, FLUX) and transformers. Core kernels: RMSNorm, RoPE 3D, GEGLU, AdaLN. Includes XCD swizzle, autotune, diffusers integration patterns, and LTX-Video pipeline injection."
disable-model-invocation: false
user-invocable: true
allowed-tools: "Read, Grep, Glob, Bash"
argument-hint: "kernel type: rmsnorm, rope, rope-3d, geglu, adaln, gemm, benchmark, diffusers, transformers, ltx-video, huggingface-kernels, get_kernel, autotune, xcd-swizzle"
---

# ROCm Triton Kernels for Diffusers & Transformers

This skill provides patterns and guidance for developing optimized Triton kernels targeting AMD GPUs (MI355X, R9700) on ROCm, for use with HuggingFace **diffusers** (LTX-Video, SD3, FLUX) and **transformers** libraries.

## Quick Start

### Diffusers (LTX-Video)

**Inject optimized kernels into LTX-Video pipeline:**
```python
import os
os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'

from diffusers import LTXPipeline
pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")  # ROCm uses same API via HIP
inject_optimized_kernels(pipe)  # BEFORE CPU offloading
pipe.enable_model_cpu_offload()
```

**For a minimal integration example (~150 lines):**
```bash
python scripts/ltx_kernel_injection_example.py
```

### Isolated Kernel Micro-benchmarks
```bash
# All 4 kernels: correctness + performance + bandwidth
python scripts/benchmark_kernels.py

# Single kernel
python scripts/benchmark_kernels.py --kernel rmsnorm
python scripts/benchmark_kernels.py --kernel rope
python scripts/benchmark_kernels.py --kernel geglu
python scripts/benchmark_kernels.py --kernel adaln
```

### End-to-End Pipeline Benchmark
```bash
# Compare baseline vs Triton vs torch.compile
python scripts/benchmark_e2e.py --mode all

# Quick test
python scripts/benchmark_e2e.py --mode triton --num-frames 9 --steps 5

# Save results for comparison
python scripts/benchmark_e2e.py --mode all --output-json results.json
```

## Target Model: LTX-Video

### Architecture Overview

| Component | Class | Has Weight | Count | Kernel |
|-----------|-------|------------|-------|--------|
| `transformer_blocks.*.norm1` | RMSNorm | **No** (elementwise_affine=False) | 56 | RMSNorm |
| `transformer_blocks.*.norm2` | RMSNorm | **No** | 56 | RMSNorm |
| `transformer_blocks.*.attn1.norm_q` | torch.nn.RMSNorm | Yes | 28 | RMSNorm |
| `transformer_blocks.*.attn1.norm_k` | torch.nn.RMSNorm | Yes | 28 | RMSNorm |
| `transformer_blocks.*.ff` | FeedForward | - | 28 | **GELU** (not GEGLU!) |
| Rotary position encoding | LTXVideoRotaryPosEmbed | - | 1 | RoPE 3D |

**Total RMSNorm modules: 168** (56 with weights, 112 without)

### Target Kernels

| Kernel | Use Case | Input Layout | Key Challenge |
|--------|----------|-------------|---------------|
| **RMSNorm** | Normalization | `[..., hidden_size]` | Weight may be None; 168 instances |
| **RoPE 3D** | Video position encoding | `[batch, t*h*w, heads, head_dim]` | 3D → temporal + spatial decomposition |
| **GEGLU** | Gated activation (SD3/FLUX) | `[batch, seq, 2*hidden]` → `[batch, seq, hidden]` | Gate/value split |
| **AdaLN** | Conditioned normalization (DiT) | `norm(x) * weight * (1+scale) + shift` | Fused norm + condition |

## Supported Hardware

| GPU | Architecture | Wave Size | LDS/CU | Mem BW | Key Feature |
|-----|-------------|-----------|--------|--------|-------------|
| **MI355X** | CDNA3+ (gfx950) | Wave64 | **160 KB** | 8 TB/s | 32 XCDs, XCD Swizzle for GEMM |
| **R9700** | RDNA4 (gfx1201) | **Wave32** | 64 KB | ~608 GB/s | 256B cacheline, inference-focused |

> See [MI355X guide](references/mi355x-optimization-guide.md) | [R9700 guide](references/r9700-optimization-guide.md)

## When This Skill Applies

Use this skill when:
- Writing Triton kernels for **RMSNorm, RoPE, GEGLU, AdaLN** on AMD GPUs
- Integrating custom kernels with **diffusers** pipelines (LTX-Video, SD3, FLUX)
- Benchmarking kernel performance against PyTorch baseline on ROCm
- Optimizing existing kernels for MI355X or R9700 architecture
- Debugging ROCm/HIP-specific kernel issues

## Critical ROCm Constraints

### Things That DON'T Work on AMD

```python
# FORBIDDEN - CUDA only, NOT available on ROCm
tl.libdevice.tanh(x)          # Use manual formula below
tl.libdevice.log1p(x)         # Use: tl.log(1.0 + x)
tl.math.tanh(x)               # Also NOT available on ROCm Triton

# Manual tanh (ONLY reliable method on ROCm):
e2x = tl.exp(2.0 * x)
tanh_x = (e2x - 1.0) / (e2x + 1.0)

# FORBIDDEN - Triton limitations on ROCm
break / continue               # Use: tl.where()
min(a, b) / max(a, b)          # Use: tl.minimum(a, b) / tl.maximum(a, b)
```

### Mandatory Environment Variables

```python
import os
os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'
```

## Core Kernel Implementations

### 1. RMSNorm (Core Optimization Target)

Row-wise reduction pattern. **168 instances** in LTX-Video, ~5% of total compute.

**CRITICAL: Do NOT autotune BLOCK_D.** Autotune may pick `BLOCK_D < D`, causing partial row processing and wrong results. Always compute `BLOCK_D = triton.next_power_of_2(D)` in the Python wrapper.

```python
@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr,
    stride_x, D,
    eps: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / D
    rms_inv = tl.rsqrt(variance + eps)

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        out = x * rms_inv * w
    else:
        out = x * rms_inv

    tl.store(out_ptr + row * stride_x + offs, out.to(x.dtype), mask=mask)


def triton_rmsnorm(x, weight=None, eps=1e-6):
    x_2d = x.contiguous().view(-1, x.shape[-1])
    out = torch.empty_like(x_2d)
    M, D = x_2d.shape
    has_weight = weight is not None
    if not has_weight:
        weight = torch.empty(0, device=x.device)

    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    rmsnorm_kernel[(M,)](
        x_2d, weight, out, x_2d.stride(0), D, eps, has_weight,
        BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=2,
    )
    return out.view_as(x)
```

**LTX-Video pitfall: Weight may be None!**
```python
has_weight = hasattr(module, 'weight') and module.weight is not None
```

### 2. RoPE 3D (Video Position Encoding)

Element-wise pattern. LTX-Video splits `head_dim` into temporal + spatial components.

**CRITICAL: cos/sin have shape `[seq_len, head_dim]`.** When grid flattens batch dimension (`batch * seq_len`), use `pid_s % seq_len` to index cos/sin, otherwise batch > 1 causes OOB GPU crash.

```python
@triton.jit
def rope_3d_kernel(
    qk_ptr, cos_ptr, sin_ptr, out_ptr,
    seq_len, num_heads, head_dim,
    stride_s, stride_h, stride_d,
    BLOCK_HD: tl.constexpr,
):
    pid_s = tl.program_id(0)  # batch * seq_len
    pid_h = tl.program_id(1)  # head index
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


def triton_rope_3d(qk, cos, sin):
    qk = qk.contiguous()
    out = torch.empty_like(qk)
    batch, seq_len, num_heads, head_dim = qk.shape
    qk_flat = qk.view(batch * seq_len, num_heads, head_dim)
    out_flat = out.view(batch * seq_len, num_heads, head_dim)
    BLOCK_HD = triton.next_power_of_2(head_dim // 2)
    num_warps = 4 if BLOCK_HD <= 64 else 8
    rope_3d_kernel[(batch * seq_len, num_heads)](
        qk_flat, cos, sin, out_flat,
        seq_len, num_heads, head_dim,
        qk_flat.stride(0), qk_flat.stride(1), qk_flat.stride(2),
        BLOCK_HD=BLOCK_HD, num_warps=num_warps, num_stages=2,
    )
    return out
```

### 3. GEGLU (For SD3/FLUX, NOT LTX-Video)

Element-wise gated activation. Input `[batch, seq, 2*hidden]` → Output `[batch, seq, hidden]`.

**Same BLOCK_SIZE rule: compute dynamically, do NOT autotune.**

```python
@triton.jit
def geglu_kernel(
    input_ptr, output_ptr,
    stride_in, stride_out, hidden_size,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < hidden_size

    gate = tl.load(input_ptr + row * stride_in + offs, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(input_ptr + row * stride_in + hidden_size + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU approx — manual tanh (tl.math.tanh NOT available on ROCm)
    k = 0.7978845608028654  # sqrt(2/pi)
    tanh_arg = k * (gate + 0.044715 * gate * gate * gate)
    e2x = tl.exp(2.0 * tanh_arg)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)
    gate_gelu = 0.5 * gate * (1.0 + tanh_val)
    result = gate_gelu * value

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
    geglu_kernel[(M,)](
        x_2d, out, x_2d.stride(0), out.stride(0), hidden_size,
        BLOCK_H=BLOCK_H, num_warps=num_warps, num_stages=2,
    )
    return out.view(*batch_dims, hidden_size)
```

**Warning: LTX-Video uses GELU, NOT GEGLU.** GEGLU is for SD3/FLUX.

### 4. AdaLN (Adaptive Layer Normalization for DiT)

Fused normalization + conditioning: `norm(x) * weight * (1 + scale) + shift`

**Same BLOCK_D rule: compute dynamically.**

```python
@triton.jit
def adaln_kernel(
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


def triton_adaln(x, weight, scale, shift, eps=1e-6):
    x_flat = x.contiguous().view(-1, x.shape[-1])
    scale_flat = scale.contiguous().view(-1, x.shape[-1])
    shift_flat = shift.contiguous().view(-1, x.shape[-1])
    out = torch.empty_like(x_flat)
    M, D = x_flat.shape
    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    adaln_kernel[(M,)](
        x_flat, weight, scale_flat, shift_flat, out,
        x_flat.stride(0), scale_flat.stride(0), D, eps,
        BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=2,
    )
    return out.view_as(x)
```

## Diffusers Integration

> **See [diffusers-integration.md](references/diffusers-integration.md) for the complete guide.**

### Minimal Integration Pattern

```python
def patch_rmsnorm_modules(model):
    """Patch all RMSNorm modules to use custom Triton kernel."""
    for name, module in model.named_modules():
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
                def make_forward(epsilon):
                    def forward(x):
                        w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
                        return triton_rmsnorm(x, w, eps=epsilon)
                    return forward
                module.forward = make_forward(eps)

pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")
patch_rmsnorm_modules(pipe.transformer)
pipe.enable_model_cpu_offload()
```

### Diffusers Critical Pitfalls

1. **RMSNorm weight may be None** — LTX-Video uses `elementwise_affine=False`
2. **Diffusers RMSNorm != torch.nn.RMSNorm** — Use `type(module).__name__` not `isinstance()`
3. **LTX-Video uses GELU, not GEGLU** — Don't patch GEGLU for LTX-Video
4. **Inject BEFORE CPU offloading** — `inject_kernels()` then `enable_model_cpu_offload()`

## Performance Expectations

### Micro-benchmark Results (MI355X, BF16)

| Kernel | Avg Speedup | Best Config Speedup | Status |
|--------|:-----------:|:-------------------:|:------:|
| **RMSNorm** | **1.71x** | 2.44x ([4×4096×3072]) | PASS |
| **RoPE 3D** | **1.21x** | 1.52x ([2×4096×16×128]) | PASS |
| **GEGLU** | **1.43x** | 2.13x ([4×4096×8192]) | PASS |
| **AdaLN** | **2.22x** | 2.77x ([4×4096×3072]) | PASS |

RMSNorm bandwidth utilization: 3554 GB/s (MI355X theoretical: 8 TB/s, ~44%).

### End-to-End LTX-Video (MI355X, 25 frames, 30 steps)

| Mode | Time (s) | Per Step (s) | Peak Mem (GB) | Speedup |
|------|:--------:|:------------:|:-------------:|:-------:|
| baseline | 1.20 | 0.040 | 18.58 | 1.00x |
| **triton** | **0.98** | **0.033** | **18.58** | **1.22x** |
| torch.compile | 0.78 | 0.026 | 18.58 | 1.54x |

**Key finding**: MI355X Triton E2E speedup (22%) is significantly higher than H100 CUDA reference (6%), because MI355X's default PyTorch RMSNorm path has more room for optimization.

### CUDA Reference (H100, for comparison)

| Shape | Custom (ms) | PyTorch (ms) | Speedup |
|:---|:---:|:---:|:---:|
| [1×1024×2048] | 0.019 | 0.065 | **3.37x** |
| [2×4096×3072] | 0.087 | 0.208 | **2.41x** |

H100 E2E: ~6% (RMSNorm is ~5% of total compute).

### Optimization Targets

| Kernel | Current | Target | Priority |
|--------|:-------:|:------:|:--------:|
| RMSNorm | 1.71x | >2x | P0 — increase bandwidth util (44%→60%+) |
| AdaLN | 2.22x | >2.5x | P1 — already strong |
| GEGLU | 1.43x | >1.5x | P1 — tanh overhead |
| RoPE 3D | 1.21x | >1.5x | P2 — small head_dim launch overhead |

## Common Issues on ROCm

| Issue | Symptom | Fix |
|-------|---------|-----|
| **Autotune BLOCK_D** | Wrong results (max_abs 4-8+) | **Never autotune BLOCK_D.** Use `triton.next_power_of_2(D)` |
| **RoPE batch OOB** | GPU crash (`Memory access fault`) | Use `pid_s % seq_len` for cos/sin indexing |
| `tl.libdevice` | Not found on AMD | Use manual math formulas |
| `tl.tanh` / `tl.math.tanh` | Not on ROCm | Manual: `e2x=exp(2x); (e2x-1)/(e2x+1)` |
| Python min/max | Runtime error | `tl.minimum()`/`tl.maximum()` |
| LDS overflow | HIP OOM | Reduce num_stages to 2 |
| Weight is None | AttributeError | Check `elementwise_affine` |
| isinstance() miss | RMSNorm not patched | Use `type(module).__name__` |

> See [troubleshooting.md](references/troubleshooting.md) for all common issues.

## Performance Profiling

```bash
rocprof --stats python your_kernel.py
rocprofv3 -i metrics.txt python your_kernel.py
rocm-bandwidth-test
rocminfo | grep -E "Name|Compute Unit|Wavefront"
```

## See Also

### Benchmark & Test Scripts
- [benchmark_kernels.py](scripts/benchmark_kernels.py) - Micro-benchmark all 4 kernels (correctness + perf + bandwidth)
- [benchmark_e2e.py](scripts/benchmark_e2e.py) - End-to-end LTX-Video pipeline benchmark (baseline vs Triton vs compile)
- [ltx_kernel_injection_example.py](scripts/ltx_kernel_injection_example.py) - Minimal diffusers injection example
- [transformers_injection_example.py](scripts/transformers_injection_example.py) - Minimal transformers injection example
- [huggingface_kernels_example.py](scripts/huggingface_kernels_example.py) - HuggingFace Kernels Hub integration example

### Integration Guides
- [diffusers-integration.md](references/diffusers-integration.md) - LTX-Video pipeline integration
- [transformers-integration.md](references/transformers-integration.md) - LLaMA/Mistral/Qwen integration
- [huggingface-kernels-integration.md](references/huggingface-kernels-integration.md) - HuggingFace Kernels Hub (`get_kernel`)
- [kernel-templates.md](references/kernel-templates.md) - Complete Triton kernel templates (incl. GEMM with XCD Swizzle)

### GPU Optimization Guides
- [mi355x-optimization-guide.md](references/mi355x-optimization-guide.md) - MI355X (gfx950) deep dive
- [r9700-optimization-guide.md](references/r9700-optimization-guide.md) - R9700 (RDNA4) deep dive

### Reference
- [troubleshooting.md](references/troubleshooting.md) - Common issues and solutions
- [kernelbench-classification.md](references/kernelbench-classification.md) - KernelBench operator taxonomy
- [skill-evaluation-methodology.md](references/skill-evaluation-methodology.md) - How to evaluate and improve skills
- [kernel-agent-knowledge-base.md](references/kernel-agent-knowledge-base.md) - Knowledge from kernel-agent project

### External Resources
- [Triton Documentation](https://triton-lang.org/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HuggingFace Kernels Hub](https://huggingface.co/kernels-community)
- [LTX-Video on HuggingFace](https://huggingface.co/Lightricks/LTX-Video)
- [HuggingFace Diffusers](https://huggingface.co/docs/diffusers/en/index)
