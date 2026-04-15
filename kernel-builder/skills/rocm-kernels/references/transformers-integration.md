# Transformers Library Integration Guide (ROCm / Triton)

Complete guide for integrating custom Triton kernels into HuggingFace transformers models on AMD GPUs.

> **Quick Start:** See [transformers_injection_example.py](../scripts/transformers_injection_example.py) for a minimal working example (~150 lines).

## Overview

The HuggingFace transformers library has different architecture patterns than diffusers. Understanding these patterns is critical for successful kernel integration with models like LLaMA, Mistral, Qwen, and other LLMs on ROCm.

**Key difference from diffusers:** All transformers RMSNorm modules have weights (`elementwise_affine=True`). No need to handle the weight-less variant.

## Model Architecture Analysis

```python
from transformers import AutoModelForCausalLM, AutoConfig
import torch

config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
print(f"Hidden size: {config.hidden_size}")       # 4096
print(f"Num layers: {config.num_hidden_layers}")   # 32
print(f"Num heads: {config.num_attention_heads}")   # 32
print(f"RMS norm eps: {config.rms_norm_eps}")       # 1e-6

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.bfloat16,
    device_map="cuda"  # ROCm uses same API via HIP
)

for name, module in model.named_modules():
    class_name = type(module).__name__
    if 'Norm' in class_name:
        has_weight = hasattr(module, 'weight') and module.weight is not None
        print(f"{name}: {class_name} (has_weight={has_weight})")
```

## Common Transformers Architectures

### LLaMA / Llama-2 / Llama-3

| Component | Class | Has Weight | Notes |
|-----------|-------|------------|-------|
| `model.norm` | LlamaRMSNorm | Yes | Final layer norm |
| `model.layers.*.input_layernorm` | LlamaRMSNorm | Yes | Pre-attention norm |
| `model.layers.*.post_attention_layernorm` | LlamaRMSNorm | Yes | Pre-FFN norm |
| `model.layers.*.mlp` | LlamaMLP | - | Uses SiLU gating |

### Mistral / Mixtral

| Component | Class | Has Weight | Notes |
|-----------|-------|------------|-------|
| `model.norm` | MistralRMSNorm | Yes | Final layer norm |
| `model.layers.*.input_layernorm` | MistralRMSNorm | Yes | Pre-attention norm |
| `model.layers.*.post_attention_layernorm` | MistralRMSNorm | Yes | Pre-FFN norm |

### Qwen / Qwen2 / Qwen3

| Component | Class | Has Weight | Notes |
|-----------|-------|------------|-------|
| `model.norm` | Qwen2RMSNorm | Yes | Final layer norm |
| `model.layers.*.input_layernorm` | Qwen2RMSNorm | Yes | Pre-attention norm |
| `model.layers.*.post_attention_layernorm` | Qwen2RMSNorm | Yes | Pre-FFN norm |

### Kernel Applicability

| Kernel | LLaMA | Mistral | Qwen | Notes |
|--------|-------|---------|------|-------|
| RMSNorm | **Yes** | **Yes** | **Yes** | All use RMSNorm with weights |
| GEGLU | No | No | No | Uses SiLU gating instead |
| RoPE | Indirect | Indirect | Indirect | Computed by transformers internally |
| Attention | Via SDPA | Via SDPA | Via SDPA | Use Flash Attention 2 |

## Integration Pattern

### Step 1: Set ROCm Environment Variables

```python
import os
os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'
```

### Step 2: Define the Triton RMSNorm Kernel

```python
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, weight_ptr, out_ptr,
    stride_x, D,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / D
    rms_inv = tl.rsqrt(variance + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    out = x * rms_inv * w

    tl.store(out_ptr + row * stride_x + offs, out.to(x.dtype), mask=mask)


def triton_rmsnorm(x, weight, eps=1e-6):
    x_2d = x.contiguous().view(-1, x.shape[-1])
    out = torch.empty_like(x_2d)
    M, D = x_2d.shape
    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    rmsnorm_fwd_kernel[(M,)](
        x_2d, weight, out, x_2d.stride(0), D, eps,
        BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=2,
    )
    return out.view_as(x)
```

### Step 3: Create RMSNorm Patcher

```python
def patch_rmsnorm_modules(model) -> int:
    """
    Patch all RMSNorm modules to use Triton kernel on ROCm.

    Works with LlamaRMSNorm, MistralRMSNorm, Qwen2RMSNorm, etc.
    """
    patched_count = 0

    for name, module in model.named_modules():
        class_name = type(module).__name__

        if 'RMSNorm' in class_name:
            # LLaMA uses 'variance_epsilon', others use 'eps'
            eps = getattr(module, 'variance_epsilon', None)
            if eps is None:
                eps = getattr(module, 'eps', 1e-6)

            has_weight = hasattr(module, 'weight') and module.weight is not None

            if has_weight:
                def make_patched_forward(mod, epsilon):
                    def patched_forward(hidden_states):
                        return triton_rmsnorm(hidden_states, mod.weight, eps=epsilon)
                    return patched_forward
                module.forward = make_patched_forward(module, eps)
                patched_count += 1
            else:
                print(f"WARNING: {name} has no weight, skipping")

    return patched_count
```

### Step 4: Use in Script

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.bfloat16,
    device_map="cuda"  # ROCm uses same device API via HIP
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

count = patch_rmsnorm_modules(model)
print(f"Patched {count} RMSNorm modules")
# Expected: 65 modules (32 layers * 2 + 1 final)

inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

## Key Differences from Diffusers

### 1. RMSNorm Always Has Weight

Unlike diffusers (where some RMSNorm modules have `elementwise_affine=False`), transformers RMSNorm modules **always** have weights. The `HAS_WEIGHT` branch is always true, so you can simplify the kernel to always load weights.

### 2. Different Epsilon Attribute Names

```python
# LLaMA uses 'variance_epsilon'
eps = getattr(module, 'variance_epsilon', 1e-6)

# Some models use 'eps'
eps = getattr(module, 'eps', 1e-6)

# Safe pattern
eps = getattr(module, 'variance_epsilon', None) or getattr(module, 'eps', 1e-6)
```

### 3. No Attention Processor Pattern

Diffusers uses `set_processor()` for attention modules. Transformers does not:

```python
# Transformers: Use Flash Attention 2 instead of custom processors
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
```

### 4. Device Map vs Manual Move

```python
# Transformers — use device_map for large models
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"  # Handles multi-GPU automatically
)

# Diffusers — manual move then CPU offload
pipe = DiffusionPipeline.from_pretrained(model_id)
pipe.to("cuda")
pipe.enable_model_cpu_offload()
```

## ROCm-Specific Considerations

### 1. ROCm Environment Setup

```python
import os
os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'
```

### 2. No tl.libdevice / tl.math.tanh

If you extend beyond RMSNorm (e.g., custom SiLU activation), remember tanh is not available:

```python
# Manual tanh for ROCm
e2x = tl.exp(2.0 * x)
tanh_x = (e2x - 1.0) / (e2x + 1.0)
```

### 3. Verify HIP Backend

```python
import torch
print(f"HIP version: {torch.version.hip}")  # Should show ROCm version
print(f"GPU: {torch.cuda.get_device_name()}")
```

### 4. torch.compile on ROCm

Custom Triton kernels and `torch.compile` can coexist on ROCm since Triton is already the compilation backend. However, test thoroughly as behavior may differ from eager mode.

## Model-Specific Integration

### LLaMA Models

```python
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

count = patch_rmsnorm_modules(model)
print(f"Patched {count} LlamaRMSNorm modules")
# Expected: 65 modules (32 layers * 2 + 1 final)
```

### Qwen3-8B

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

count = patch_rmsnorm_modules(model)
print(f"Patched {count} Qwen2RMSNorm modules")
# Expected: 65 modules (32 layers * 2 + 1 final)
```

## Verification

### Verify Injection Worked

```python
x = torch.randn(1, 10, model.config.hidden_size, device='cuda', dtype=torch.bfloat16)
for name, module in model.named_modules():
    if 'RMSNorm' in type(module).__name__:
        out = module(x)
        print(f"RMSNorm forward pass: {x.shape} -> {out.shape}")
        break
```

### Run Generation Test

```python
inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

### Profile on ROCm

```bash
rocprof --stats python your_script.py
rocprofv3 -i metrics.txt python your_script.py
```

## Performance Optimization

### Enable Flash Attention 2

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda"
)
```

### Combine with Custom Kernels

```python
model = AutoModelForCausalLM.from_pretrained(model_id, ...)
patch_rmsnorm_modules(model)  # Inject Triton RMSNorm
# Flash Attention 2 handles attention optimization
```
