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

Hub kernels must be loaded with either a `version` or an explicit `revision`.

> [!NOTE]
> Version `0` kernels are excluded from the API compatibility requirement,
> since it is used for alpha/beta-quality kernels that may still have
> rapidly changing APIs.

## Checking Kernel Availability

You can check if a particular version of a kernel supports the environment
that the program is running on:

```python
from kernels import has_kernel

# Check if kernel is available for current environment
is_available = has_kernel("kernels-community/activation", version=1)
print(f"Kernel available: {is_available}")
```

When no compatible kernel is found, [`~kernels.has_kernel`] does not say _why_.
[`~kernels.get_kernel_variants`] returns the full resolution trace instead: one
decision per build variant in the repository, with compatible variants listed
first. Each decision is a `VariantAccepted` or a `VariantRejected`, and rejected
variants carry a human-readable `reason`:

```python
from kernels import get_kernel_variants, VariantAccepted

for decision in get_kernel_variants("kernels-community/activation", version=1):
    name = decision.variant.variant_str
    if isinstance(decision, VariantAccepted):
        print(f"{name}: compatible")
    else:
        print(f"{name}: rejected ({decision.reason})")
```

## Inspecting Loaded Kernels

[`~kernels.get_loaded_kernels`] returns a snapshot of every kernel that has been loaded
into the current process. Each entry is a [`~kernels.LoadedKernel`] with the kernel
`metadata`, the imported `module`, and a `repo_info` ([`~kernels.RepoInfo`] with the
`repo_id`, resolved `revision`, and a `local` flag).

```python
from kernels import get_kernel, get_loaded_kernels

get_kernel("kernels-community/activation", version=1)

for loaded in get_loaded_kernels():
    print(loaded.metadata.name, loaded.repo_info)
```

`repo_info` is populated for kernels loaded from a known Hub repository, i.e. via
[`~kernels.get_kernel`], [`~kernels.get_locked_kernel`], or [`~kernels.load_kernel`].
Only [`~kernels.get_local_kernel`], which loads from an arbitrary local path, leaves
`repo_info=None`. Its `local` flag distinguishes kernels downloaded from the Hub
(`local=False`) from pre-downloaded, locked kernels loaded from local files
(`local=True`).

Browse through different kernels compatible with `kernels` from [here](https://huggingface.co/kernels).

A kernel can provide layers in addition to kernel functions. Refer to [Layers](./layers.md) to know more.
