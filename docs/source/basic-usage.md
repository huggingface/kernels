# Basic Usage

## Loading Kernels

Here is how you would use the [activation](https://huggingface.co/kernels-community/activation) kernels from the Hugging Face Hub:

```python
import torch
from kernels import get_kernel

# Download optimized kernels from the Hugging Face hub
activation = get_kernel("kernels-community/activation")

# Create a random tensor
x = torch.randn((10, 10), dtype=torch.float16, device="cuda")

# Run the kernel
y = torch.empty_like(x)
activation.gelu_fast(y, x)

print(y)
```

### Using version bounds

Kernels are versioned using tags of the form `v<major>.<minor>.<patch>`.
You can specify which version to download using Python version specifiers:

```python
import torch
from kernels import get_kernel

activation = get_kernel("kernels-community/activation", version=">=0.0.4,<0.1.0")
```

This will get the latest kernel tagged `v0.0.z` where `z` is at least 4. It
is strongly recommended to specify a version bound, since a kernel author
might push incompatible changes to the `main` branch.

## Checking Kernel Availability

You can check if a specific kernel is available for your environment:

```python
from kernels import has_kernel

# Check if kernel is available for current environment
is_available = has_kernel("kernels-community/activation")
print(f"Kernel available: {is_available}")
```
