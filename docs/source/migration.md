# Migrate from older versions

## 0.12

### Adopting kernel versions

Before `kernels` 0.12, kernels could be pulled from a repository
without specifying a version. This is deprecated in kernels 0.12
and is an error in kernels 0.15. Instead, use of a kernel should
always specify a version or revision (except for local kernels).

Kernels only use a major version. The kernel maintainer is responsible
for never breaking a kernel within a major version and should bump up
the major version if the kernel API changes and/or when support for
older Torch versions is removed.

> [!NOTE]
> Version `0` kernels are excluded from the API compatibility requirement,
> since it is used for alpha/beta-quality kernels that may still have
> rapidly changing APIs.

You can find the versions that are supported by a kernel using the
`kernels versions command`. For example:

```bash
$ kernels versions kernels-community/activation
Version 1: torch210-cxx11-cu126-x86_64-linux, torch210-cxx11-cu128-x86_64-linux, torch210-cxx11-cu130-x86_64-linux, torch27-cxx11-cu118-x86_64-linux, torch27-cxx11-cu126-x86_64-linux, torch27-cxx11-cu128-aarch64-linux, torch27-cxx11-cu128-x86_64-linux ✅, torch28-cxx11-cu126-aarch64-linux, torch28-cxx11-cu126-x86_64-linux, torch28-cxx11-cu128-aarch64-linux, torch28-cxx11-cu128-x86_64-linux, torch28-cxx11-cu129-aarch64-linux, torch28-cxx11-cu129-x86_64-linux, torch29-cxx11-cu126-aarch64-linux, torch29-cxx11-cu126-x86_64-linux, torch29-cxx11-cu128-aarch64-linux, torch29-cxx11-cu128-x86_64-linux, torch29-cxx11-cu130-aarch64-linux, torch29-cxx11-cu130-x86_64-linux
```

The command lists all available versions (here only version 1) with
all the variants that are supported. A check mark is printed after
the variant that is compatible with your current environment.

Code that uses a kernel can be updated as follows:

```python
# Old:
activation = get_kernel("kernels-community/activation")
activation = get_kernel("kernels-community/activation", version=">=0.0.2 && <0.1.0")

# New:
activation = get_kernel("kernels-community/activation", version=1)

# Old:
kernel_layer_mapping = {
    "SiluAndMul": {
        "cuda": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
        ),
    }
}
kernel_layer_mapping = {
    "SiluAndMul": {
        "cuda": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
            version=">=0.0.2 && <0.1.0",
        ),
    }
}

# New:
kernel_layer_mapping = {
    "SiluAndMul": {
        "cuda": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
            version=1,
        ),
    }
}
```

## 0.14

### `kernel` repo type on the Hub

Kernels are now a first-class repository type on the Hugging Face Hub, and
`kernels` 0.14 loads kernels exclusively from `kernel`-type repositories.
`model`-type kernel repositories are no longer supported by the loader.

New uploads via `kernel-builder build-and-upload` default to
`--repo-type kernel`. To publish, the owning user or org must have
kernel-creation access. Request it from
[huggingface.co/settings/account](https://huggingface.co/settings/account)
("Request Kernels Creation").

To migrate an existing `model`-type kernel repository:

1. Make sure the publishing org has been granted kernel-creation access
   (see above).
2. Re-upload with `kernel-builder build-and-upload` to a `kernel`-type
   repository. Either keep the same `repo-id` in `build.toml` if the
   repository has been migrated to the new type, or point it at a newly
   created `kernel`-type repository.
3. Update consumers' [`~kernels.get_kernel`] and [`~kernels.LayerRepository`] calls
   to reference the new repository if the `repo-id` changed.

## 0.16

> [!WARNING] **Deprecation of kernel functions.**
> `use_kernel_func_from_hub`, `FuncRepository`, `LocalFuncRepository`, and
`LockedFuncRepository` are now deprecated.

To make a function extensible by a layer, use the same decorator as for
layers ([`~kernels.use_kernel_forward_from_hub`]). This makes it clearer
that the function is actually replaced by a layer. You can also use the
[`~kernels.use_kernelized_func`] decorator to attach such a function to
the layer where it is used, making it discoverable by
[`~kernels.kernelize`].

For example:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import use_kernel_func_from_hub  # old
from kernels import use_kernel_forward_from_hub, use_kernelized_func

# Old:
@use_kernel_func_from_hub("silu_and_mul")
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]

class FeedForward(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.silu_and_mul = silu_and_mul
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu_and_mul(self.linear(x))

# New:
@use_kernel_forward_from_hub("silu_and_mul")
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]

@use_kernelized_func(silu_and_mul)
class FeedForward(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(self.linear(x))
```

`FuncRepository`, `LocalFuncRepository`, and `LockedFuncRepository` are
not replaced. They allowed using an arbitrary function from a kernel
as a layer, but this was easily misused and did not have a clean way
of marking such a function as supporting `torch.compile` or backward
passes. Going forward, kernel functions should be exposed as regular
kernel layers and used with [`~kernels.LayerRepository`],
[`~kernels.LocalLayerRepository`], or [`~kernels.LockedLayerRepository`].
For example:

```python
import torch
import torch.nn as nn

# Old:
def fast_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
  ...

# New:
def fast_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
  ...

# Kernel layer that exposes the function.
class FastSiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fast_silu_and_mul(x)
```

For more information, see the [layer documentation](layers.md).
