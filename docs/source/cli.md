# Kernels CLI Reference

The `kernels` CLI provides commands for managing compute kernels.

```bash
kernels [-h] {check,download,versions,upload,lock,generate-readme,benchmark,init} ...
```

## Commands

| Command                                   | Description                                              |
| ----------------------------------------- | -------------------------------------------------------- |
| [init](cli-init.md)                       | Initialize a new kernel project from template            |
| [upload](cli-upload.md)                   | Upload kernels to the Hub                                |
| [benchmark](cli-benchmark.md)             | Run benchmark results for a kernel                       |
| [check](cli-check.md)                     | Check a kernel for compliance                            |
| [versions](cli-versions.md)               | Show kernel versions                                     |
| [generate-readme](cli-generate-readme.md) | Generate README snippets for a kernel's public functions |
| [lock](cli-lock.md)                       | Lock kernel revisions                                    |
| [download](cli-download.md)               | Download locked kernels                                  |

## Quick Start

### Create a new kernel project

```bash
kernels init my-username/my-kernel
cd my-kernel
```

### Build and test locally

```bash
cachix use huggingface
nix run -L --max-jobs 1 --cores 8 .#build-and-copy
uv run example.py
```

### Upload to the Hub

```bash
kernels upload ./build --repo-id my-username/my-kernel
```

### Use kernels in your project

#### Directly from the Hub

```python
import torch

from kernels import get_kernel

# Download optimized kernels from the Hugging Face hub
my_kernel = get_kernel("my-username/my-kernel", version=1)

# Random tensor
x = torch.randn((10, 10), dtype=torch.float16, device="cuda")

# Run the kernel
y = torch.empty_like(x)
my_kernel.my_kernel_function(y, x)

print(y)
```

or 

#### Locked and downloaded

Add to `pyproject.toml`:

```toml
[tool.kernels.dependencies]
"my-username/my-kernel" = "1"
```

Then lock and download:

```bash
kernels lock .
kernels download .
```
