# HuggingFace Kernels Integration Guide

Complete guide for using and publishing CUDA kernels with the HuggingFace Kernels library (`get_kernel`).

> **Quick Start:** See [huggingface_kernels_example.py](../scripts/huggingface_kernels_example.py) for a minimal working example.

## Overview

The [HuggingFace Kernels](https://huggingface.co/docs/kernels/en/index) library enables dynamic loading of pre-compiled CUDA kernels from the Hugging Face Hub. This eliminates the need for local compilation and ensures compatibility across different Python, PyTorch, and CUDA versions.

**Key Benefits:**
- **No local compilation** - Download pre-built binaries
- **Version management** - Load specific kernel versions
- **Multi-version support** - Multiple versions coexist in one Python process
- **Automatic compatibility** - Matches your PyTorch/CUDA configuration

## Installation

```bash
pip install kernels torch numpy
```

Requirements:
- PyTorch >= 2.5
- CUDA-capable GPU
- Python 3.8+

## Core API

### get_kernel

Download and load a kernel from the Hub:

```python
from kernels import get_kernel

# Basic usage
kernel = get_kernel("kernels-community/activation")

# With specific version
kernel = get_kernel("kernels-community/activation", version=1)

# With specific revision (branch/tag/commit)
kernel = get_kernel("kernels-community/flash-attn", revision="v2.0.0")
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_id` | str | required | Hub repository (e.g., "kernels-community/activation") |
| `revision` | str | "main" | Branch, tag, or commit hash |
| `version` | int/str | None | Kernel version number (mutually exclusive with `revision`) |
| `user_agent` | str/dict | None | Telemetry information |

**Returns:** `ModuleType` - the imported kernel module

### has_kernel

Check if a kernel build exists for your environment:

```python
from kernels import has_kernel

if has_kernel("kernels-community/activation"):
    kernel = get_kernel("kernels-community/activation")
else:
    print("No compatible build available")
```

### get_local_kernel

Load a kernel from a local path (useful for development):

```python
from kernels import get_local_kernel

# Load from local directory
kernel = get_local_kernel("/path/to/my-kernel")
```

### load_kernel & get_locked_kernel

For reproducible, offline-capable deployments using lockfiles:

```python
from kernels import load_kernel, get_locked_kernel

# Load using a lockfile
kernel = load_kernel("lockfile.json")

# Get kernel with lock
kernel = get_locked_kernel("kernels-community/activation", lockfile="kernel.lock")
```

## Usage Examples

### 1. Basic Activation Kernel

```python
import torch
from kernels import get_kernel

# Load activation kernels from Hub
activation = get_kernel("kernels-community/activation", version=1)

# Create test tensor
x = torch.randn((10, 10), dtype=torch.float16, device="cuda")

# Execute kernel (output tensor must be pre-allocated)
y = torch.empty_like(x)
activation.gelu_fast(y, x)

print(y)
```

### 2. Flash Attention

```python
import torch
from kernels import get_kernel

flash_attn = get_kernel("kernels-community/flash-attn")

# Check available functions
print(dir(flash_attn))

# Usage depends on specific kernel API
```

### 3. RMSNorm Kernel

```python
import torch
from kernels import get_kernel

layer_norm = get_kernel("kernels-community/triton-layer-norm")

# Apply RMSNorm
x = torch.randn(2, 1024, 2048, dtype=torch.bfloat16, device="cuda")
weight = torch.ones(2048, dtype=torch.bfloat16, device="cuda")
out = layer_norm.rms_norm(x, weight, eps=1e-6)
```

### 4. Integration with Transformers Models

```python
import torch
import torch.nn as nn
from kernels import get_kernel

# Load RMSNorm kernel
rmsnorm_kernel = get_kernel("kernels-community/triton-layer-norm")

def patch_rmsnorm_with_hub_kernel(model):
    """Patch model's RMSNorm to use Hub kernel."""
    for name, module in model.named_modules():
        if 'RMSNorm' in type(module).__name__:
            eps = getattr(module, 'variance_epsilon', None) or getattr(module, 'eps', 1e-6)

            def make_forward(mod, epsilon):
                def forward(hidden_states):
                    return rmsnorm_kernel.rms_norm(hidden_states, mod.weight, eps=epsilon)
                return forward

            module.forward = make_forward(module, eps)
```

### 5. Integration with Diffusers Pipelines

```python
import torch
from diffusers import LTXPipeline
from kernels import get_kernel, has_kernel

# Load kernel if available
if has_kernel("kernels-community/activation"):
    activation = get_kernel("kernels-community/activation")

    def patch_activations(model):
        # Patch GELU activations with optimized kernel
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.GELU):
                def make_forward():
                    def forward(x):
                        out = torch.empty_like(x)
                        activation.gelu_fast(out, x)
                        return out
                    return forward
                module.forward = make_forward()

# Use with pipeline
pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")
patch_activations(pipe.transformer)
```

## Publishing Kernels to Hub

### Project Structure

```
my-kernel/
├── build.toml           # Build configuration
├── kernel_src/
│   └── my_kernel.cu     # CUDA source
├── torch-ext/
│   ├── torch_binding.cpp
│   ├── torch_binding.h
│   └── my_kernel/
│       └── __init__.py
└── flake.nix           # Optional: reproducible builds
```

### build.toml Configuration

```toml
[general]
name = "my_kernel"
backends = ["cuda"]

[torch]
src = [
  "torch-ext/torch_binding.cpp",
  "torch-ext/torch_binding.h"
]

[kernel.my_kernel]
backend = "cuda"
src = ["kernel_src/my_kernel.cu"]
depends = ["torch"]

# Target specific GPU architectures
cuda-capabilities = ["7.5", "8.0", "9.0"]  # T4, A100, H100
```

### Torch Bindings

**torch_binding.h:**
```cpp
#pragma once
#include <torch/torch.h>

void my_kernel_forward(torch::Tensor &out, torch::Tensor const &input);
```

**torch_binding.cpp:**
```cpp
#include <torch/library.h>
#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("my_kernel_forward(Tensor! out, Tensor input) -> ()");
  ops.impl("my_kernel_forward", torch::kCUDA, &my_kernel_forward);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
```

### Python Wrapper

**torch-ext/my_kernel/__init__.py:**
```python
from typing import Optional
import torch
from ._ops import ops

def forward(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply my custom kernel."""
    if out is None:
        out = torch.empty_like(x)
    ops.my_kernel_forward(out, x)
    return out
```

### Building and Publishing

**Using kernel-builder (Nix):**
```bash
# Build for all platforms
nix run github:huggingface/kernel-builder -- build

# Push to Hub
huggingface-cli upload my-username/my-kernel ./dist
```

**Using pip (local development):**
```bash
pip install kernel-builder
kernel-builder build
```

## Available Community Kernels

Popular kernels from `kernels-community`:

| Kernel | Description | Usage |
|--------|-------------|-------|
| `activation` | GELU, SiLU, etc. | `get_kernel("kernels-community/activation")` |
| `flash-attn` | Flash Attention 2 | `get_kernel("kernels-community/flash-attn")` |
| `triton-layer-norm` | LayerNorm, RMSNorm | `get_kernel("kernels-community/triton-layer-norm")` |
| `quantization` | INT8/INT4 ops | `get_kernel("kernels-community/quantization")` |

Browse all kernels: https://huggingface.co/kernels-community

## Inspecting Kernel Functions

Kernel function signatures vary by implementation. Always inspect before use:

```python
from kernels import get_kernel

kernel = get_kernel("kernels-community/activation")

# List available functions
print(dir(kernel))
# ['gelu_fast', 'gelu_new', 'silu', ...]

# Check function signature (if available)
import inspect
print(inspect.signature(kernel.gelu_fast))
```

## Caching and Offline Usage

Downloaded kernels are cached in the HuggingFace Hub cache directory:
- Default: `~/.cache/huggingface/hub/`
- Override: Set `HF_HOME` environment variable

For offline usage:
```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"

# Will only use cached kernels
kernel = get_kernel("kernels-community/activation")
```

## Best Practices

1. **Check availability first:**
   ```python
   if has_kernel("kernels-community/my-kernel"):
       kernel = get_kernel("kernels-community/my-kernel")
   else:
       # Fallback to PyTorch implementation
   ```

2. **Pin versions for reproducibility:**
   ```python
   kernel = get_kernel("kernels-community/activation", version=1)
   ```

3. **Use lockfiles for production:**
   ```python
   kernel = load_kernel("kernel.lock")
   ```

4. **Pre-allocate output tensors:**
   ```python
   # Most kernels require pre-allocated outputs
   out = torch.empty_like(x)
   kernel.function(out, x)
   ```

5. **Test with your exact environment:**
   ```python
   # Print environment info
   import torch
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA: {torch.version.cuda}")
   print(f"GPU: {torch.cuda.get_device_name()}")
   ```

## Troubleshooting

### No compatible build found

```python
from kernels import has_kernel, get_kernel

if not has_kernel("kernels-community/my-kernel"):
    print("No build for your PyTorch/CUDA version")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
```

### Import errors after loading

```python
# Always inspect available functions
kernel = get_kernel("kernels-community/activation")
print(dir(kernel))  # Check what's actually available
```

### Version conflicts

```python
# Explicitly specify version
kernel_v1 = get_kernel("repo/kernel", version=1)
kernel_v2 = get_kernel("repo/kernel", version=2)
# Both can coexist in the same process
```

## See Also

- [HuggingFace Kernels Documentation](https://huggingface.co/docs/kernels/en/index)
- [HuggingFace Kernels GitHub](https://github.com/huggingface/kernels)
- [Kernel Builder Documentation](https://github.com/huggingface/kernel-builder)
- [Community Kernels](https://huggingface.co/kernels-community)
- [Blog: Learn the Kernel Hub in 5 Minutes](https://huggingface.co/blog/hello-hf-kernels)
