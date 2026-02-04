# Basic Usage

## Loading Kernels

Here is how you would use the [relu](https://huggingface.co/kernels-community/relu) kernels from the Hugging Face Hub:

```python
import torch
from kernels import get_kernel

# Download optimized kernels from the Hugging Face hub
relu = get_kernel("kernels-community/relu", version=1)

# Create a random tensor
x = torch.randn((10, 10), dtype=torch.float, device="cuda")

# Run the kernel
y = torch.empty_like(x)
relu.relu(x, y)

print(y)
```

This fetches version `1` of the kernel `kernels-community/relu`.
Kernels are versioned using a major version number. Using `version=1` will
get the latest kernel build from the `v1` branch.

Kernels within a version branch must never break the API or remove builds
for older PyTorch versions. This ensures that your code will continue to work.

Some kernels have not yet been updated to use versioning yet. In these cases,
you can use `get_kernel` without the `version` argument.

## Checking Kernel Availability

You can check if a particular version of a kernel supports the environment
that the program is running on:

```python
from kernels import has_kernel

# Check if kernel is available for current environment
is_available = has_kernel("kernels-community/relu", version=1)
print(f"Kernel available: {is_available}")
```
