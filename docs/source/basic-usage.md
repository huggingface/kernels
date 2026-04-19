# Quickstart

## Loading Kernels

Here is how you would use the [activation](https://huggingface.co/kernels-community/activation) kernels from the Hugging Face Hub:

```python
import torch
from kernels import get_kernel

# Download optimized kernels from the Hugging Face hub
activation = get_kernel("kernels-community/activation", version=1)

# Create a random tensor
x = torch.randn((10, 10), dtype=torch.float16, device="cuda")

# Run the kernel
y = torch.empty_like(x)
activation.gelu_fast(y, x)

print(y)
```

This fetches version `1` of the kernel `kernels-community/activation`.
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
is_available = has_kernel("kernels-community/activation", version=1)
print(f"Kernel available: {is_available}")
```

## Inspecting Loaded Kernels

`get_loaded_kernels()` returns a snapshot of every kernel that has been loaded
into the current process. Each entry is a `LoadedKernel` namedtuple with the
imported `module`, the `package_name`, and `repo_infos` (repo id, resolved
revision, and the backend argument that was passed).

```python
from kernels import get_kernel, get_loaded_kernels

get_kernel("kernels-community/activation", version=1)

for loaded in get_loaded_kernels():
    print(loaded.package_name, loaded.repo_infos)
```

`repo_infos` is populated only for kernels loaded with `get_kernel`. Kernels
loaded from a local path (`get_local_kernel`) or via a lockfile
(`get_locked_kernel`, `load_kernel`) have `repo_infos=None`.
