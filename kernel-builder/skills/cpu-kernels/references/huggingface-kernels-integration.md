# HuggingFace Kernels Integration Guide (CPU)

Complete guide for using and publishing CPU kernels with the HuggingFace Kernels library.

## Overview

The [HuggingFace Kernels](https://huggingface.co/docs/kernels/en/index) library enables dynamic loading of pre-compiled kernels from the Hugging Face Hub. CPU kernels are compiled via `kernel-builder` and distributed as platform-specific wheels.

**Key Benefits:**
- **No local compilation** — download pre-built binaries
- **Version management** — load specific kernel versions
- **Multi-backend** — same API for CUDA, XPU, and CPU kernels
- **Automatic compatibility** — matches your PyTorch configuration

## Installation

```bash
pip install kernels torch
```

## Core API

### get_kernel

```python
from kernels import get_kernel

kernel = get_kernel("kernels-community/rmsnorm")

# With specific version
kernel = get_kernel("kernels-community/rmsnorm", version=1)
```

### has_kernel

```python
from kernels import has_kernel

if has_kernel("kernels-community/rmsnorm"):
    kernel = get_kernel("kernels-community/rmsnorm")
```

### get_local_kernel (Development)

```python
from pathlib import Path
from kernels import get_local_kernel

# Load from local build (requires Path object and package name)
kernel = get_local_kernel(Path("/path/to/my-kernel"), "my_kernel")
```

## CPU Kernel Usage

### RMSNorm Example

```python
import torch
from kernels import get_kernel, has_kernel

repo_id = "kernels-community/rmsnorm"

if has_kernel(repo_id):
    layer_norm = get_kernel(repo_id)

    x = torch.randn(2, 1024, 2048, dtype=torch.bfloat16)  # CPU tensor
    weight = torch.ones(2048, dtype=torch.bfloat16)

    # CPU dispatch happens automatically via torch_binding.cpp
    out = layer_norm.rms_norm_fn(x, weight, eps=1e-6)
```

### Integration with Transformers

```python
import torch
from kernels import get_kernel, has_kernel

repo_id = "kernels-community/rmsnorm"

if has_kernel(repo_id):
    rmsnorm_kernel = get_kernel(repo_id)

    def patch_rmsnorm(model):
        """Patch model's RMSNorm to use CPU kernel."""
        patched = 0
        for name, module in model.named_modules():
            if 'RMSNorm' in type(module).__name__:
                eps = getattr(module, 'variance_epsilon', None) or getattr(module, 'eps', 1e-6)

                def make_forward(mod, epsilon):
                    def forward(hidden_states):
                        return rmsnorm_kernel.rms_norm(hidden_states, mod.weight, eps=epsilon)
                    return forward

                module.forward = make_forward(module, eps)
                patched += 1
        return patched
```

## Publishing a CPU Kernel

### 1. Build with kernel-builder

```bash
cd my-kernel/
kernel-builder build --release
# Produces dist/my_kernel-*.whl
```

### 2. Test locally

```bash
pip install dist/*.whl --force-reinstall
python -c "from pathlib import Path; from kernels import get_local_kernel; k = get_local_kernel(Path('.'), 'my_kernel'); print(dir(k))"
```

### 3. Create Hub repository

```bash
# Create repo on huggingface.co/kernels-community/
# Upload the kernel source and build.toml
```

### 4. Multi-backend support

A single kernel repo can support CUDA, XPU, and CPU. The `build.toml` defines all backends:

```toml
# CUDA sections
[kernel.my_kernel_cuda]
backend = "cuda"
# ...

# CPU sections
[kernel.my_kernel_cpu]
backend = "cpu"
# ...
```

The kernel-builder CI builds wheels for each backend. `get_kernel()` automatically selects the right wheel for the user's hardware.

## Kernel File Layout (Hub)

```
my-kernel/
├── build.toml              # Multi-target build config
├── torch-ext/
│   └── torch_binding.cpp   # Op registration (registration.h)
├── my_kernel_cpu/           # CPU implementation
│   ├── cpu_features.hpp
│   ├── my_kernel_cpu.cpp
│   ├── my_kernel_cpu.hpp
│   ├── my_kernel_cpu_torch.cpp
│   ├── my_kernel_avx512.cpp
│   └── my_kernel_avx512.hpp
├── csrc/                    # CUDA implementation (if any)
│   └── ...
└── README.md
```

## Notes

- CPU kernels use the same `torch_binding.cpp` and `registration.h` pattern as CUDA/XPU kernels
- The `ops.impl("forward", torch::kCPU, &func)` call ensures CPU dispatch
- Multi-device kernels use `#if defined(CPU_KERNEL)` / `#elif defined(CUDA_KERNEL)` guards
- CPU wheels are built for x86_64 Linux; ARM/macOS may require source builds
