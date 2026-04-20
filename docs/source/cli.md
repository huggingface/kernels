# Kernels CLI Reference

The `kernels` CLI provides commands for managing compute kernels.

## Commands

| Command                                                 | Description                                              |
| ------------------------------------------------------- | -------------------------------------------------------- |
| [upload](cli-upload.md)                                 | Upload kernels to the Hub                                |
| [benchmark](cli-benchmark.md)                           | Run benchmark results for a kernel                       |
| [check](cli-check.md)                                   | Check a kernel for compliance                            |
| [versions](cli-versions.md)                             | Show kernel versions                                     |
| [lock](cli-lock.md)                                     | Lock kernel revisions                                    |
| [download](cli-download.md)                             | Download locked kernels                                  |
| [skills](cli-skills-add.md)                             | Add skills for AI coding assistants                      |

## Quick Start

For building and writing kernels, please refer [building kernels](./builder/build.md) and 
[writing kernels](./builder/writing-kernels.md).

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

### See help

```bash
kernels --help
```