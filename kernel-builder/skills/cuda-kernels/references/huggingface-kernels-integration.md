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

# Load a specific major version (the standard way). A bare
# get_kernel(repo_id) raises ValueError — version= or revision= is required.
kernel = get_kernel("kernels-community/activation", version=1)

# Or pin an explicit revision (branch/tag/commit). This is for exceptional cases, using `version` is strongly recommended.
kernel = get_kernel("kernels-community/flash-attn", revision="v2.0.0")
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_id` | str | required | Hub repository (e.g., "kernels-community/activation") |
| `version` | int | None | Kernel major version — **one of `version` or `revision` is required** |
| `revision` | str | None | Branch, tag, or commit hash (mutually exclusive with `version`) |
| `user_agent` | str/dict | None | Telemetry information |

**Returns:** `ModuleType` - the imported kernel module

### has_kernel

Check if a kernel build exists for your environment:

```python
from kernels import has_kernel

if has_kernel("kernels-community/activation", version=1):
    kernel = get_kernel("kernels-community/activation", version=1)
else:
    print("No compatible build available")
```

### get_local_kernel

Load a locally built kernel (useful for development — no Hub access, no version needed). Pass the kernel **project root**; it resolves variants from `<path>` or `<path>/build`:

```python
from pathlib import Path
from kernels import get_local_kernel

# Load the freshly built kernel (after `kernel-builder build-and-copy -L`)
kernel = get_local_kernel(Path("/path/to/my-kernel"))
```

Alternatively, the `LOCAL_KERNELS` environment variable redirects `get_kernel()` itself to a local build — production integration code can then be tested **unchanged**:

```bash
LOCAL_KERNELS="my-username/my-kernel=/path/to/my-kernel" python app.py
# get_kernel("my-username/my-kernel") now loads the local build/
# (the override skips the Hub, trust checks, and version resolution)
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

flash_attn = get_kernel("kernels-community/flash-attn", version=1)

# Check available functions
print(dir(flash_attn))

# Usage depends on specific kernel API
```

### 3. RMSNorm Kernel

```python
import torch
from kernels import get_kernel

layer_norm = get_kernel("kernels-community/triton-layer-norm", version=1)

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
rmsnorm_kernel = get_kernel("kernels-community/triton-layer-norm", version=1)

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
if has_kernel("kernels-community/activation", version=1):
    activation = get_kernel("kernels-community/activation", version=1)

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

Scaffold new projects with `kernel-builder init` — it generates this layout (plus `benchmarks/`, `example.py`) with a valid `build.toml` and an initialized git repository:

```bash
kernel-builder init --name my-username/my-kernel
```

```
my-kernel/
├── build.toml           # Build configuration
├── flake.nix            # Required: kernel-builder's Nix build entry point
├── CARD.md              # Kernel card template (uploaded as README.md)
├── my_kernel_cuda/
│   └── my_kernel.cu     # CUDA source (any dir name; listed in build.toml src)
├── torch-ext/
│   ├── torch_binding.cpp
│   ├── torch_binding.h
│   └── my_kernel/
│       └── __init__.py
└── tests/
    └── test_my_kernel.py
```

### build.toml Configuration

```toml
[general]
# Dash-separated lowercase name (underscores are rejected by check-config);
# the Python package dir is torch-ext/<name with dashes -> underscores>.
name = "my-kernel"
backends = ["cuda"]
version = 1
license = "Apache-2.0"

[general.hub]
# Hub repository to upload to (used by `kernel-builder build-and-upload`);
# together with `version` this selects the version branch (e.g. v1).
repo-id = "my-username/my-kernel"

[torch]
src = [
  "torch-ext/torch_binding.cpp",
  "torch-ext/torch_binding.h"
]

[kernel.my_kernel]
backend = "cuda"
src = ["my_kernel_cuda/my_kernel.cu"]
depends = ["torch"]

# Leave cuda-capabilities unspecified unless the kernel truly requires
# specific architectures — over-constraining produces non-compliant builds.
# cuda-capabilities = ["9.0", "10.0", "12.0"]
```

### Torch Bindings

> This is the **only** supported binding pattern. Do NOT use pybind11 (`PYBIND11_MODULE`, `#include <torch/extension.h>`) or `TORCH_LIBRARY` with a hardcoded namespace — both fail under kernel-builder's ABI3 build. See "Hard Constraints" in SKILL.md.

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

### Layers (Optional)

A kernel can also export `torch.nn.Module` layers that `kernels.kernelize()` swaps in for matching model layers. Per the [kernel requirements](https://huggingface.co/docs/kernels/kernel-requirements), layers must be **pure**: subclass `torch.nn.Module`, define no constructor, no class variables (except `has_backward` / `can_torch_compile`), and no methods other than `forward`. Put them in `torch-ext/my_kernel/layers.py` and export the module from the main `__init__.py`:

```python
# torch-ext/my_kernel/layers.py
import torch
from ._ops import ops

class MyKernelLayer(torch.nn.Module):
    has_backward: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        ops.my_kernel_forward(out, x)
        return out
```

A layer's `forward` may also use member variables (e.g. `weight`, `bias`) that are defined by the layer it extends. Since the layer defines no constructor, these are not assigned here — but their types can be annotated as class-level type hints purely for type checking:

```python
# torch-ext/my_kernel/layers.py
import torch
from ._ops import ops

class RMSNorm(torch.nn.Module):
    has_backward: bool = True
    can_torch_compile: bool = True

    # Defined by the layer being extended; annotated for type checking.
    weight: torch.Tensor
    variance_epsilon: float

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # `rms_norm` is defined by the kernel.
        return ops.rms_norm(
            hidden_states,
            self.weight,
            self.variance_epsilon,
        )
```

```python
# torch-ext/my_kernel/__init__.py
from . import layers

__all__ = [..., "layers"]
```

### Building and Publishing

**Using kernel-builder (Nix):**
```bash
# Build the kernel locally (run inside the kernel directory)
kernel-builder build-and-copy -L

# Build and upload to the Hub in one go
kernel-builder build-and-upload
```

The target repository is determined by the `repo-id` and `version` fields in `build.toml` (see above). Uploads go to a **`kernel`-type** Hub repository (the first-class kernel repository type), not a model repo — the owning user or org needs kernel-creation access, requested via [huggingface.co/settings/account](https://huggingface.co/settings/account) ("Request Kernels Creation"). If a `CARD.md` template is present in the source repo, it is filled and uploaded as the `README.md`.

**Editable install for local development** (never hand-write a `setup.py` — `torch.utils.cpp_extension`/pybind11 cannot build under ABI3):
```bash
kernel-builder create-pyproject -f
pip install wheel
pip install --no-build-isolation -e .
```

## Available Community Kernels

Popular kernels from `kernels-community`:

| Kernel | Description | Usage |
|--------|-------------|-------|
| `activation` | GELU, SiLU, etc. | `get_kernel("kernels-community/activation", version=1)` |
| `flash-attn` | Flash Attention 2 | `get_kernel("kernels-community/flash-attn", version=1)` |
| `triton-layer-norm` | LayerNorm, RMSNorm | `get_kernel("kernels-community/triton-layer-norm", revision="main")` |
| `quantization` | INT8/INT4 ops | `get_kernel("kernels-community/quantization", revision="main")` |

Browse all kernels: https://huggingface.co/kernels-community

## Inspecting Kernel Functions

Kernel function signatures vary by implementation. Always inspect before use:

```python
from kernels import get_kernel

kernel = get_kernel("kernels-community/activation", version=1)

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
kernel = get_kernel("kernels-community/activation", version=1)
```

## Best Practices

1. **Check availability first:**
   ```python
   if has_kernel("kernels-community/my-kernel", version=1):
       kernel = get_kernel("kernels-community/my-kernel", version=1)
   else:
       # Fallback to PyTorch implementation
   ```

2. **Always pass `version=` (it is required, not optional):**
   ```python
   kernel = get_kernel("kernels-community/activation", version=1)
   ```
   Version branches (`v1`, `v2`, ...) never break the kernel API, so pinning the major version keeps code working while still receiving fixes.

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

if not has_kernel("kernels-community/my-kernel", version=1):
    print("No build for your PyTorch/CUDA version")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
```

### Import errors after loading

```python
# Always inspect available functions
kernel = get_kernel("kernels-community/activation", version=1)
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
