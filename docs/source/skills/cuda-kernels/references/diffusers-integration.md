# Diffusers Pipeline Integration Guide

Complete guide for integrating custom CUDA kernels into HuggingFace diffusers pipelines.

> **Quick Start:** See [ltx_kernel_injection_example.py](ltx_kernel_injection_example.py) for a minimal working example (~150 lines).

## Overview

Diffusers pipelines (LTX-Video, Stable Diffusion, FLUX, DiT) have specific architecture patterns. Understanding these patterns is critical for successful kernel integration.

## Model Architecture Analysis

Before integrating kernels, analyze the target model:

```python
# 1. Check pipeline components
from diffusers import LTXPipeline
import inspect

pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
print("Components:", [k for k in dir(pipe) if not k.startswith('_') and hasattr(getattr(pipe, k), 'named_modules')])
# Output: ['transformer', 'vae', 'text_encoder']

# 2. Analyze transformer structure
for name, module in pipe.transformer.named_modules():
    class_name = type(module).__name__
    if 'Norm' in class_name or 'Attention' in class_name or 'GELU' in class_name:
        has_weight = hasattr(module, 'weight') and module.weight is not None
        print(f"{name}: {class_name} (has_weight={has_weight})")
```

## LTX-Video Architecture

### Key Components

| Component | Class | Has Weight | Notes |
|-----------|-------|------------|-------|
| `transformer_blocks.*.norm1` | RMSNorm | **No** | elementwise_affine=False |
| `transformer_blocks.*.norm2` | RMSNorm | **No** | elementwise_affine=False |
| `transformer_blocks.*.attn1.norm_q` | torch.nn.RMSNorm | Yes | elementwise_affine=True |
| `transformer_blocks.*.attn1.norm_k` | torch.nn.RMSNorm | Yes | elementwise_affine=True |
| `transformer_blocks.*.ff` | FeedForward | - | Uses GELU (not GEGLU!) |

### Kernel Applicability

| Kernel | Used in LTX-Video | Notes |
|--------|-------------------|-------|
| RMSNorm | **Yes** | 168 modules (56 with weights, 112 without) |
| GEGLU | **No** | LTX uses GELU with tanh approximation |
| RoPE 3D | Indirect | Diffusers computes its own via LTXVideoRotaryPosEmbed |
| AdaLN | Partial | Scale/shift pattern in transformer blocks |

## Integration Pattern

### Step 1: Create Optimized Attention Processor

```python
from typing import Optional, Tuple
import torch
from ltx_kernels import rmsnorm

class OptimizedLTXVideoAttnProcessor:
    """
    Custom attention processor using optimized CUDA kernels.

    Replaces RMSNorm operations for Q/K normalization with custom kernel.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Import here to avoid circular imports
        from diffusers.models.transformers.transformer_ltx import apply_rotary_emb
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Q, K, V projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Custom RMSNorm kernel for Q/K normalization
        # NOTE: attn.norm_q and attn.norm_k HAVE weights (elementwise_affine=True)
        query = rmsnorm(query, attn.norm_q.weight, eps=attn.norm_q.eps)
        key = rmsnorm(key, attn.norm_k.weight, eps=attn.norm_k.eps)

        # Apply rotary embeddings (computed by diffusers)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Reshape for multi-head attention
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Dispatch attention (PyTorch SDPA or other backends)
        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
```

### Step 2: Create Module Patcher

```python
import torch.nn as nn
from ltx_kernels import rmsnorm

def patch_rmsnorm_modules(model: nn.Module) -> int:
    """
    Patch all RMSNorm modules to use custom CUDA kernel.

    Handles both:
    - Modules WITH weight (elementwise_affine=True)
    - Modules WITHOUT weight (elementwise_affine=False)

    Returns:
        Number of modules patched.
    """
    patched_count = 0

    for name, module in model.named_modules():
        # Check by class name (not isinstance) to catch diffusers RMSNorm
        if type(module).__name__ == 'RMSNorm':
            eps = getattr(module, 'eps', 1e-6)
            has_weight = hasattr(module, 'weight') and module.weight is not None

            if has_weight:
                def make_patched_forward_with_weight(mod, epsilon):
                    def patched_forward(x):
                        return rmsnorm(x, mod.weight, eps=epsilon)
                    return patched_forward
                module.forward = make_patched_forward_with_weight(module, eps)
            else:
                # No weight (elementwise_affine=False) - use ones
                def make_patched_forward_no_weight(epsilon):
                    def patched_forward(x):
                        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
                        return rmsnorm(x, weight, eps=epsilon)
                    return patched_forward
                module.forward = make_patched_forward_no_weight(eps)

            patched_count += 1

    return patched_count
```

### Step 3: Create Injection Function

```python
def inject_optimized_kernels(pipe) -> dict:
    """
    Inject custom CUDA kernels into the pipeline.

    Call this AFTER loading model to CUDA, BEFORE enabling CPU offloading.

    Returns:
        dict with counts of patched modules.
    """
    stats = {
        'attention_processors': 0,
        'rmsnorm_modules': 0,
    }

    if not hasattr(pipe, 'transformer'):
        print("WARNING: Pipeline has no 'transformer' attribute!")
        return stats

    transformer = pipe.transformer

    # 1. Replace attention processors
    for name, module in transformer.named_modules():
        if hasattr(module, 'set_processor') and hasattr(module, 'processor'):
            module.set_processor(OptimizedLTXVideoAttnProcessor())
            stats['attention_processors'] += 1

    # 2. Patch RMSNorm modules
    stats['rmsnorm_modules'] = patch_rmsnorm_modules(transformer)

    return stats
```

### Step 4: Use in Script

```python
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

# Import kernels
from ltx_kernels import rmsnorm

# Load pipeline
pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Inject kernels BEFORE CPU offloading
stats = inject_optimized_kernels(pipe)
print(f"Attention processors replaced: {stats['attention_processors']}")
print(f"RMSNorm modules patched: {stats['rmsnorm_modules']}")

# Now enable memory optimization
pipe.enable_model_cpu_offload()

# Generate video
output = pipe(
    prompt="A cat sleeping in the sun",
    num_frames=25,
    height=480,
    width=704,
    num_inference_steps=30,
)

export_to_video(output.frames[0], "output.mp4", fps=24)
```

## Model-Specific Differences

### LTX-Video
- Uses GELU (not GEGLU)
- RMSNorm in blocks has no weight
- RMSNorm in attention has weight
- Custom 3D RoPE computed by diffusers

### Stable Diffusion 3
- Uses GEGLU in FeedForward
- May have different normalization patterns
- Check before assuming kernel applicability

### FLUX
- Uses GEGLU
- Different attention patterns
- Verify architecture before patching

## Verification

### Verify Injection Worked

```python
# Check attention processors
for name, module in pipe.transformer.named_modules():
    if hasattr(module, 'processor'):
        print(f"{name}: {type(module.processor).__name__}")
        break
# Should show: OptimizedLTXVideoAttnProcessor

# Test a forward pass
with torch.inference_mode():
    x = torch.randn(1, 100, 2048, device='cuda', dtype=torch.bfloat16)
    for name, module in pipe.transformer.named_modules():
        if type(module).__name__ == 'RMSNorm':
            out = module(x)
            print(f"RMSNorm forward pass: {x.shape} -> {out.shape}")
            break
```

### Run Full Inference Test

```bash
.venv/bin/python generate_video.py --num-frames 9 --steps 5
# Quick test with minimal frames/steps
```

## Troubleshooting

See SKILL.md "Common Issues and Solutions" for:
- Weight is None errors
- isinstance() not working
- GEGLU not being called
- CPU offloading issues

## Complete Example

For a self-contained, runnable example that demonstrates all patterns above:

```bash
cd examples/ltx_video
uv pip install -e .  # Build kernels
python ../../.claude/skills/h100-diffusers-kernels/references/ltx_kernel_injection_example.py
```

This example:
1. Loads LTX-Video pipeline
2. Injects custom kernels
3. Verifies injection worked
4. Generates a short test video
