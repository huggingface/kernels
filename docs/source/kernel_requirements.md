# Kernel requirements

Kernels on the Hub must fulfill the requirements outlined on this page.
You can use [kernel-builder](https://github.com/huggingface/kernel-builder/)
to build conforming kernels.

## Directory layout

A kernel repository on the Hub must contain a `build` directory. This
directory contains build variants of a kernel in the form of directories
following the template
`<framework><version>-cxx<abiver>-<cu><cudaver>-<arch>-<os>`.
For example `build/torch26-cxx98-cu118-x86_64-linux`. The currently
recommended build variants are:

- `torch25-cxx11-cu118-x86_64-linux`
- `torch25-cxx11-cu121-x86_64-linux`
- `torch25-cxx11-cu124-x86_64-linux`
- `torch25-cxx98-cu118-x86_64-linux`
- `torch25-cxx98-cu121-x86_64-linux`
- `torch25-cxx98-cu124-x86_64-linux`
- `torch26-cxx11-cu118-x86_64-linux`
- `torch26-cxx11-cu124-x86_64-linux`
- `torch26-cxx11-cu126-x86_64-linux`
- `torch26-cxx98-cu118-x86_64-linux`
- `torch26-cxx98-cu124-x86_64-linux`
- `torch26-cxx98-cu126-x86_64-linux`

This list will be updated as new PyTorch versions are released. Kernels
that are in pure Python (e.g. Triton kernels) only need to provide a
single build variant:

- `torch-universal`

Each variant directory should contain a single directory with the same name
as the repository (replacing `-` by `_`). For instance, kernels in the
`kernels-community/activation` repository have a directories like
`build/<variant>/activation`. This directory
must be a Python package with an `__init__.py` file.

## Versioning

Kernels are versioned on the Hub using Git tags. Version tags must be of
the form `v<major>.<minor>.<patch>`. Versions are used by [locking](./locking.md)
to resolve the version constraints.

## Native Python module

Kernels will typically contain a native Python module with precompiled
compute kernels and bindings. This module must fulfill the following
requirements:

- Use [ABI3/Limited API](https://docs.python.org/3/c-api/stable.html#stable-application-binary-interface)
  for compatibility with Python 3.9 and later.
- Compatible with [`manylinux_2_28`](https://github.com/pypa/manylinux?tab=readme-ov-file#manylinux_2_28-almalinux-8-based).
  This means that the extension **must not** use symbols versions higher than:

  - GLIBC 2.28
  - GLIBCXX 3.4.24
  - CXXABI 1.3.11
  - GCC 7.0.0

  These requirement can be checked with the ABI checker (see below).

- No dynamic library dependencies outside:

  - Torch;
  - CUDA/ROCm libraries installed as dependencies of Torch.

The manylinux_2_28 and Python ABI 3.9 version requirements can be checked with
[`kernel-abi-check`](https://crates.io/crates/kernel-abi-check):

```bash

$ cargo install kernel-abi-check
$ kernel-abi-check result/relu/_relu_e87e0ca_dirty.abi3.so
🐍 Checking for compatibility with manylinux_2_28 and Python ABI version 3.9
✅ No compatibility issues found
```

## Torch extension

Torch native extension functions must be [registered](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial)
in `torch.ops.<namespace>`. Since we allow loading of multiple versions of
a module in the same Python process, `namespace` must be unique for each
version of a kernel. Failing to do so will create clashes when different
versions of the same kernel are loaded. Two suggested ways of doing this
are:

- Appending a truncated SHA-1 hash of the git commit that the kernel was
  built from to the name of the extension.
- Appending random material to the name of the extension.

**Note:** we recommend against appending a version number or git tag.
Version numbers are typically not bumped on each commit, so users
might use two different commits that happen to have the same version
number. Git tags are not stable, so they do not provide a good way
of guaranteeing uniqueness of the namespace.

## Layers

A kernel can provide layers in addition to kernel functions. A layer from
the Hub can replace the `forward` method of an existing layer for a certain
device type. This makes it possible to provide more performant kernels for
existing layers. See the [layers documentation](layers.md) for more information
on how to use layers.

### Writing layers

To make the extension of layers safe, the layers must fulfill the following
requirements:

- The layers are subclasses of `torch.nn.Module`.
- The layers are pure, meaning that they do not have their own state. This
  means that:
  - The layer must not define its own constructor.
  - The layer must not use class variables.
- No other methods must be defined than `forward`.
- The `forward` method has a signature that is compatible with the
  `forward` method that it is extending.

The only exception to the _no class variables rule_ is addition of a
`has_backward` class variable. This variable is used to indicate whether
the layer has a backward pass implemented (`True` when absent).

This is an example of a pure layer:

```python
class SiluAndMul(nn.Module):
    # This layer does not implement backward.
    has_backward: bool = False

    def forward(self, x: torch.Tensor):
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.silu_and_mul(out, x)
        return out
```

For some layers, the `forward` method has to use state from the adopting class.
In these cases, we recommend to use type annotations to indicate what member
variables are expected. For instance:

```python
class LlamaRMSNorm(nn.Module):
    weight: torch.Tensor
    variance_epsilon: float

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_norm_fn(
            hidden_states,
            self.weight,
            bias=None,
            residual=None,
            eps=self.variance_epsilon,
            dropout_p=0.0,
            prenorm=False,
            residual_in_fp32=False,
        )
```

This layer expects the adopting layer to have `weight` and `variance_epsilon`
member variables and uses them in the `forward` method.

### Exporting layers

To accommodate portable loading, `layers` must be defined in the main
`__init__.py` file. For example:

```python
from . import layers

__all__ = [
  # ...
  "layers"
  # ...
]
```

## Python requirements

- Python code must be compatible with Python 3.9 and later.
- All Python code imports from the kernel itself must be relative. So,
  for instance if in the example kernel `example`,
  `module_b` needs a function from `module_a`, import as:

  ```python
  from .module_a import foo
  ```

  **Never use:**

  ```python
  # DO NOT DO THIS!

  from example.module_a import foo
  ```

  The latter would import from the module `example` that is in Python's
  global module dict. However, since we allow loading multiple versions
  of a module, we uniquely name the module.

- Only modules from the Python standard library, Torch, or the kernel itself
  can be imported.
