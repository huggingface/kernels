# Kernel requirements

Kernels on the Hub must fulfill the requirements outlined on this page. By
ensuring kernels are compliant, they can be used on a wide range of Linux
systems and Torch builds.

[Join us on GitHub Discussions](https://github.com/huggingface/kernels/discussions/categories/kernel-builder)
for questions and discussions about building kernels!

## Repository type

Compliant kernels are published as `kernel`-type repositories on the Hub
(the first-class kernel repository type). New uploads via `kernel-builder`
default to this type; see the [migration guide](migration.md) if you
maintain an older `model`-type kernel repository.

## Trusted publishers

`kernels` only loads kernels from a curated set of trusted publishers by
default. Loading from any other publisher raises an error unless the caller
opts in with `trust_remote_code=True`:

```python
# Trusted publisher: works without opt-in.
get_kernel("kernels-community/activation", version=1)

# Untrusted publisher: must opt in explicitly.
get_kernel("some-other-org/my-kernel", version=1, trust_remote_code=True)
```

The Hub also exposes a `trustedKernelPublisher` flag on the kernel API and
displays a corresponding badge in the UI.

## Directory layout

A kernel repository on the Hub must contain a `build` directory. This
directory contains build variants of a kernel in the form of directories
following the template
`<framework><version>-cxx<abiver>-<cu><cudaver>-<arch>-<os>`.
For example `build/torch26-cxx98-cu118-x86_64-linux`.

The kernel is in the build variant directory and must contain a
`__init__.py` file. For compatibility with older versions of the
`kernels` package, each variant directory must also contain a single
directory with the same name as the repository (replacing `-` by `_`).
For instance, kernels in the `kernels-community/activation` repository
have a directory like `build/<variant>/activation`. This directory
must contain an `__init__.py` file that exports the same symbols as
`__init__.py` in the build variant directory `build/<variant>`.
[This example](https://huggingface.co/kernels-test/flattened-build/blob/main/build/torch-universal/flattened_build/__init__.py)
shows how this can be done. This compatibility directory is
automatically created by `kernel-builder`.

## Build variants

A kernel can be compliant for a specific compute framework (e.g. CUDA) or
architecture (e.g. x86_64). For compliance with a compute framework and
architecture combination, all the variants from the [build variant list](builder/build-variants)
must be available for that combination.

## Kernel metadata

The build variant directory must contain a `metadata.json` file with kernel
metadata. Currently the following top-level keys are supported:

- `id` (`str`, required): a unique identifier for the kernel. This
  identifier must also be a valid Python module name. If the kernel
  registers Torch ops, they must be registered as `torch.ops.<id>`
- `name` (`str`, required): then name of the kernel. Replacing dashes
  by underscores should result in the module name of the kernel.
- `version` (`int`, required): the kernel version number.
- `license` (`str`, required): the kernel license in. Refer to the
  list of [supported license identifiers](https://huggingface.co/docs/hub/repositories-licenses).
- `upstream` (`str`, optional): Git-compatible URL (passable to `git clone`)
  of the original upstream repository where the kernel source code comes from.
- `source` (`str`, optional): Git-compatible URL (passable to `git clone`)
  of the kernel-builder formatted source repository (must contain `build.toml`
  and `flake.nix`).
- `backend` (`dict`, required): information about the compute backend that
  this build variant supports.
- `digest` (`Digest`, required): hash digest of the kernel files.
- `python-depends` (`list[str]`, optional): list of Python dependencies
  from a curated set of Python dependencies.
- `provenance` (`dict`, optional): provenance of the build, used to flag
  non-reproducible (dirty) builds. It contains two optional sub-objects:
  - `kernel-builder`: the `kernel-builder` that produced the build, with its
    `version` (`str`), the `sha` (`str`) of the `kernel-builder` source it was
    built from (when known), and a `dirty` (`bool`) flag that is `true` when
    `kernel-builder` was built from a source tree with uncommitted changes.
  - `kernel`: the kernel source that was built, with its commit `sha` (`str`)
    and a `dirty` (`bool`) flag that is `true` when the kernel source had
    uncommitted changes.

  When either `dirty` flag is set, the kernel was built from uncommitted
  sources and cannot be reliably reproduced.

  > **Note:** For Nix builds, dirtiness follows Nix's flake tree status, which
  > also counts **untracked** files (including an uncommitted `flake.lock`).
  > Commit your `flake.lock` (and avoid stray untracked files) so that clean
  > builds are not flagged as dirty. Local `create-pyproject` runs only
  > consider changes to tracked files.

  > **Note:** The kernel `sha`/`dirty` are captured at the moment
  > `create-pyproject` runs, so they describe the source tree as it was *then*.
  > Running `create-pyproject` and committing afterwards is bad practice: the
  > recorded provenance keeps pointing at the pre-commit state (a stale `sha`,
  > and `dirty: true` if the tree was dirty) even though the committed source
  > differs. Generate the metadata from the final, committed source instead.

Example `metadata.json`:

```json
{
  "name": "mykernel",
  "id": "_mykernel_cuda_7a4e5a7",
  "version": 1,
  "license": "Apache-2.0",
  "python-depends": ["einops"],
  "backend": {
    "type": "cuda",
    "archs": ["7.0", "7.2", "7.5", "8.0", "8.6", "8.7", "8.9", "9.0+PTX"]
  },
  "digest": {
    "algorithm": "sha256",
    "files": {
      "__init__.py": "xLMbARTcTl8L/m1kJLc/h/QL4Kzt772F872a46pfRGI=",
      "_mykernel_cuda_7645816_dirty.abi3.so": "vtdzzToloH38HZkVs7sFEf69QFDxROuPsBAond3Jic0=",
      "_ops.py": "Hrp5aF4o0eHSttw4sQGsbBAXFqvLJ42Y9YJ2KkqvZhg=",
      "mykernel/__init__.py": "DFYPlrhXwYjEqCl/8n0SmWGZV8NFml5DPhMjKfv98GY="
    }
  }
}
```

The `metadata.json` file is generated automatically by `kernel-builder`.

## Backend

The `backend` specifies a dictionary of the following form:

```json
{
  # ...
  "backend": {
    "type": "cuda",
    "archs": ["7.0", "7.2", "7.5", "8.0", "8.6", "8.7", "8.9", "9.0+PTX"]
  }
}
```

The backend `type` must be one of `cann`, `cpu`, `cuda`, `metal`, `neuron`,
`rocm`, or `xpu`. For CUDA and ROCm, the supported architectures must
be specified in the `archs` field.

### Python dependencies

You can specify Python dependencies that your kernel requires. Dependencies can be either general (required for all backends) or backend-specific (required only for certain compute backends like CUDA, ROCm, XPU, Metal, or CPU).

#### General dependencies

For dependencies required regardless of the backend, use the `python-depends` field:

```json
{
  "python-depends": ["einops"]
}
```

#### Backend-specific dependencies

For dependencies that are only needed for specific backends, use the `python-depends-backends` field:

```json
{
  "python-depends-backends": {
    "cuda": ["nvidia-cutlass-dsl"],
    "xpu": ["onednn"]
  }
}
```

#### Combined example

You can specify both general and backend-specific dependencies:

```json
{
  "python-depends": ["einops"],
  "python-depends-backends": {
    "cuda": ["nvidia-cutlass-dsl"]
  },
  "version": 1
}
```

#### Allowed dependencies

The following dependencies are currently allowed:

**General dependencies:**

- `einops`

**Backend-specific dependencies:**

- CUDA: `nvidia-cutlass-dsl`
- XPU: `onednn`

Dependencies are validated based on the backend being used. When a kernel is loaded, only the dependencies relevant to the active backend are checked.

## Versioning

Kernels are versioned using a major version. The kernel revisions of a
version are stored in a branch of the form `v<version>`. Each build
variant will also have the kernel version in `metadata.json`.

The version **must** be bumped in the following cases:

- The kernel API is changed in an incompatible way.
- The API is extended in a compatible way, but not all build variants
  receive the extension (e.g. because they are for older Torch versions
  that are not supported by `kernel-builder` anymore).

In both cases, build variants that are not updated must be removed from
the new version's branch.

> [!IMPORTANT]
> The *kernel API* covered by these versioning guarantees is only the
> public API: the symbols listed in the `__all__` of the kernel's
> top-level `__init__.py`. Anything not in `__all__` (e.g. internal
> helpers or names prefixed with `_`) is considered private and may
> change or be removed at any time without a version bump. Export every
> symbol you intend consumers to rely on in `__all__`.

> [!NOTE]
> By convention, we reserve version `0` for kernels that are still in
> alpha or beta stage and are not recommended for production use (e.g.
> because the API is still changing regularly or there are still too
> many issues).

## Native Python module

Kernels will typically contain a native Python module with precompiled
compute kernels and bindings. This module must fulfill the requirements
outlined in this section. For all operating systems, a kernel must not
have dynamic library dependencies outside:

- Torch;
- CUDA/ROCm libraries installed as dependencies of Torch.

## Compatibility with torch.compile

The Kernel Hub also encourages to write the kernels in a `torch.compile`
compliant way. This helps to ensure that the kernels are compatible with
`torch.compile` without introducing any graph breaks and triggering
recompilation which can limit the benefits of compilation.

[Here](https://github.com/huggingface/kernels/blob/f83b4da6b7f6b171b47bb9bf96271ae2273bc9d3/builder/examples/relu-backprop-compile/tests/test_relu.py#L162)
is a simple test example which checks for graph breaks and
recompilation triggers during `torch.compile`.

### Linux

- Use [ABI3/Limited API](https://docs.python.org/3/c-api/stable.html#stable-application-binary-interface)
  for compatibility with Python 3.9 and later.
- Compatible with [`manylinux_2_28`](https://github.com/pypa/manylinux?tab=readme-ov-file#manylinux_2_28-almalinux-8-based).
  This means that the extension **must not** use symbols versions higher than:
  - GLIBC 2.28
  - GLIBCXX 3.4.24
  - CXXABI 1.3.11
  - GCC 7.0.0

These requirements can be checked with the ABI checker (see below).

### macOS

- Use [ABI3/Limited API](https://docs.python.org/3/c-api/stable.html#stable-application-binary-interface)
  for compatibility with Python 3.9 and later.
- macOS deployment target 15.0.
- Metal 3.0 (`-std=metal3.0`).

The ABI3 requirement can be checked with the ABI checker (see below).

### ABI checker

The manylinux_2_28 and Python ABI 3.9 version requirements can be checked with
`kernel-builder check-abi`:

```bash
$ kernel-builder check-abi examples/kernels/relu
🐍 Checking for compatibility with manylinux_2_28 and Python ABI version 3.9: /home/daniel/git/kernels/examples/kernels/relu/result/torch211-cpu-x86_64-linux/_relu_cpu_30dc0ae_dirty.abi3.so
✅ No compatibility issues found
🐍 Checking for compatibility with manylinux_2_28 and Python ABI version 3.9: /home/daniel/git/kernels/examples/kernels/relu/result/torch211-cu126-x86_64-linux/_relu_cuda_30dc0ae_dirty.abi3.so
✅ No compatibility issues found
🐍 Checking for compatibility with manylinux_2_28 and Python ABI version 3.9: /home/daniel/git/kernels/examples/kernels/relu/result/torch211-cu128-x86_64-linux/_relu_cuda_30dc0ae_dirty.abi3.so
✅ No compatibility issues found
🐍 Checking for compatibility with manylinux_2_28 and Python ABI version 3.9: /home/daniel/git/kernels/examples/kernels/relu/result/torch211-cu130-x86_64-linux/_relu_cuda_30dc0ae_dirty.abi3.so
✅ No compatibility issues found
[...]
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

There are two exceptions to the _no class variables rule_:

1. The `has_backward` variable can be used to indicate whether the layer has
   a backward pass implemented (`True` when absent).
2. The `can_torch_compile` variable can be used to indicate whether the layer
   supports `torch.compile` (`False` when absent).

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

> [!IMPORTANT]
> Only symbols listed in `__all__` are treated as the kernel's public
> API. This is the surface that consumers can rely on and that the
> [versioning guarantees](#versioning) apply to, so be sure to export
> every function, class, and `layers` module you want to expose.

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
