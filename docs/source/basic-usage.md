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

## Device Architecture Checks

A kernel build declares the device architectures it was built for (e.g. CUDA
compute capabilities such as `8.0` or `9.0a`, or ROCm archs such as `gfx90a`)
in its metadata. [`~kernels.get_kernel`] validates these architectures against
the current device and raises an error when the device is not supported. This
check can be disabled by passing `check_arch=False` to [`~kernels.get_kernel`].
Without this check, an incompatible kernel would load fine, but fail at
launch time — some kernels even terminate the whole process rather than
raising an exception.

[`~kernels.has_kernel`] performs the same validation and returns `False` when
the kernel does not support the current device, so you can select a different
kernel or fall back to another implementation without hard-failing.

The declared architectures are the union over all of a kernel's components, so
a kernel may support more architectures than it declares. For example, a
kernel could fall back to a Triton implementation on architectures that its
CUDA kernels were not built for. For such kernels, the check can be disabled
with `check_arch=False`:

```python
from kernels import get_kernel, has_kernel

# Only checks that a compatible build variant exists, the kernel may not
# support the device architecture.
has_kernel("kernels-community/flash-attn3", version=1, check_arch=False)

# Skips the device architecture check when loading.
flash_attn3 = get_kernel("kernels-community/flash-attn3", version=1, check_arch=False)
```

> [!WARNING]
> When the architecture check is disabled, loading an incompatible kernel
> succeeds, but using it can fail or even terminate the process at run time.

For kernels that are used through layers, prefer
[registering kernels for specific CUDA capabilities](./layers.md#registering-kernels-for-specific-cuda-capabilities)
instead — `kernels` will then never load an incompatible kernel at all.

## Inspecting Loaded Kernels

[`~kernels.get_loaded_kernels`] returns a snapshot of every kernel that has been loaded
into the current process. Each entry is a [`~kernels.LoadedKernel`] namedtuple with the
imported `module`, the `package_name`, and `repo_infos` (repo id, resolved
revision, and the backend argument that was passed).

```python
from kernels import get_kernel, get_loaded_kernels

get_kernel("kernels-community/activation", version=1)

for loaded in get_loaded_kernels():
    print(loaded.package_name, loaded.repo_infos)
```

`repo_infos` is populated only for kernels loaded with [`~kernels.get_kernel`]. Kernels
loaded from a local path ([`~kernels.get_local_kernel`]) or via a lockfile
([`~kernels.get_locked_kernel`], [`~kernels.load_kernel`]) have `repo_infos=None`.

Browse through different kernels compatible with `kernels` from [here](https://huggingface.co/kernels).

A kernel can provide layers in addition to kernel functions. Refer to [Layers](./layers.md) to know more.
