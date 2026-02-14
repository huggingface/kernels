# Transformers Library Integration Guide

Complete guide for integrating custom CUDA kernels into HuggingFace transformers models.

> **Quick Start:** See [transformers_injection_example.py](../scripts/transformers_injection_example.py) for a minimal working example (~120 lines).

## Overview

The HuggingFace transformers library has different architecture patterns than diffusers. Understanding these patterns is critical for successful kernel integration with models like LLaMA, Mistral, Qwen, and other LLMs.

## Model Architecture Analysis

Before integrating kernels, analyze the target model:

```python
from transformers import AutoModelForCausalLM, AutoConfig
import torch

# 1. Check model configuration
config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
print(f"Hidden size: {config.hidden_size}")
print(f"Num layers: {config.num_hidden_layers}")
print(f"Num heads: {config.num_attention_heads}")
print(f"RMS norm eps: {config.rms_norm_eps}")

# 2. Load model and analyze structure
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 3. Find normalization modules
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
| `model.layers.*.mlp` | MistralMLP | - | Uses SiLU gating |

### Qwen / Qwen2

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
| RoPE | Indirect | Indirect | Indirect | Computed by transformers |
| Attention | Via SDPA | Via SDPA | Via SDPA | Use Flash Attention 2 |

## Integration Pattern

### Step 1: Create RMSNorm Patcher

```python
import torch
import torch.nn as nn
from ltx_kernels import rmsnorm

def patch_rmsnorm_modules(model: nn.Module) -> int:
    """
    Patch all RMSNorm modules to use custom CUDA kernel.

    Works with:
    - LlamaRMSNorm
    - MistralRMSNorm
    - Qwen2RMSNorm
    - Any module with 'RMSNorm' in class name

    Returns:
        Number of modules patched.
    """
    patched_count = 0

    for name, module in model.named_modules():
        class_name = type(module).__name__

        # Match all RMSNorm variants
        if 'RMSNorm' in class_name:
            # Get epsilon (different models use different attribute names)
            eps = getattr(module, 'variance_epsilon', None)
            if eps is None:
                eps = getattr(module, 'eps', 1e-6)

            # All transformers RMSNorm modules have weights
            has_weight = hasattr(module, 'weight') and module.weight is not None

            if has_weight:
                def make_patched_forward(mod, epsilon):
                    def patched_forward(hidden_states):
                        return rmsnorm(hidden_states, mod.weight, eps=epsilon)
                    return patched_forward
                module.forward = make_patched_forward(module, eps)
                patched_count += 1
            else:
                print(f"WARNING: {name} has no weight, skipping")

    return patched_count
```

### Step 2: Create Injection Function

```python
def inject_optimized_kernels(model) -> dict:
    """
    Inject custom CUDA kernels into a transformers model.

    Call this AFTER loading model to CUDA.

    Args:
        model: HuggingFace transformers model

    Returns:
        dict with counts of patched modules.
    """
    stats = {
        'rmsnorm_modules': 0,
    }

    # Patch RMSNorm modules
    stats['rmsnorm_modules'] = patch_rmsnorm_modules(model)

    return stats
```

### Step 3: Use in Script

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import kernels
from ltx_kernels import rmsnorm

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Inject kernels
stats = inject_optimized_kernels(model)
print(f"RMSNorm modules patched: {stats['rmsnorm_modules']}")

# Generate text
inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

## Key Differences from Diffusers

### 1. RMSNorm Always Has Weight

Unlike diffusers (where some RMSNorm modules have `elementwise_affine=False`), transformers RMSNorm modules **always** have weights:

```python
# Diffusers - must check for weight
has_weight = hasattr(module, 'weight') and module.weight is not None

# Transformers - weight always exists (but still good practice to check)
if hasattr(module, 'weight') and module.weight is not None:
    # Safe to use module.weight
```

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
# Diffusers pattern (NOT applicable to transformers)
# module.set_processor(CustomProcessor())

# Transformers: Use Flash Attention 2 instead
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"  # Built-in optimization
)
```

### 4. Device Map vs Manual Move

```python
# Transformers - use device_map for large models
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"  # Handles multi-GPU automatically
)

# Diffusers - manual move then CPU offload
pipe = DiffusionPipeline.from_pretrained(model_id)
pipe.to("cuda")
pipe.enable_model_cpu_offload()
```

## Model-Specific Integration

### LLaMA Models

```python
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# Patch RMSNorm
count = patch_rmsnorm_modules(model)
print(f"Patched {count} LlamaRMSNorm modules")
# Expected: 65 modules (32 layers * 2 + 1 final)
```

### Mistral Models

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

count = patch_rmsnorm_modules(model)
print(f"Patched {count} MistralRMSNorm modules")
# Expected: 65 modules (32 layers * 2 + 1 final)
```

### Qwen Models

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)

count = patch_rmsnorm_modules(model)
print(f"Patched {count} Qwen2RMSNorm modules")
```

## Verification

### Verify Injection Worked

```python
# Test forward pass
x = torch.randn(1, 10, model.config.hidden_size, device='cuda', dtype=torch.bfloat16)
for name, module in model.named_modules():
    if 'RMSNorm' in type(module).__name__:
        out = module(x)
        print(f"RMSNorm forward pass: {x.shape} -> {out.shape}")
        break

# Verify kernel is being called (check with profiler)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    _ = model.generate(inputs, max_new_tokens=10)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Run Generation Test

```python
# Quick test
inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
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

### Use torch.compile

```python
# Compile the model for faster inference
model = torch.compile(model, mode="reduce-overhead")
```

### Combine with Custom Kernels

```python
# Note: Custom kernels need torch.library registration for torch.compile
# See SKILL.md "torch.compile Compatibility" section
model = AutoModelForCausalLM.from_pretrained(model_id, ...)
inject_optimized_kernels(model)  # Inject first
# model = torch.compile(model)  # Requires custom op registration
```

## Troubleshooting

### Common Issues

1. **AttributeError: 'variance_epsilon'**
   ```python
   # Different models use different attribute names
   eps = getattr(module, 'variance_epsilon', None) or getattr(module, 'eps', 1e-6)
   ```

2. **CUDA out of memory**
   ```python
   # Use device_map for automatic sharding
   model = AutoModelForCausalLM.from_pretrained(
       model_id,
       device_map="auto",  # Spreads across available GPUs
       torch_dtype=torch.bfloat16
   )
   ```

3. **Slow first inference**
   ```python
   # Warmup run to compile CUDA kernels
   with torch.inference_mode():
       _ = model.generate(warmup_inputs, max_new_tokens=1)
   ```

4. **torch.compile errors with custom kernels**
   - Custom kernels and torch.compile are mutually exclusive without custom op registration
   - Either use custom kernels OR torch.compile, not both

## Complete Example

For a self-contained, runnable example:

```bash
cd examples/ltx_video
uv pip install -e .  # Build kernels
python ../../.claude/skills/h100-diffusers-kernels/scripts/transformers_injection_example.py
```

This example:
1. Loads a transformers model (LLaMA/Mistral)
2. Injects custom RMSNorm kernel
3. Verifies injection worked
4. Runs generation benchmark
