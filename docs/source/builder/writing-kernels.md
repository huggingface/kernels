# Writing Hub kernels with kernel-builder

## Introduction

The Kernel Hub allows Python libraries and applications to load compute
kernels directly from the [Hub](https://hf.co/). To support this kind
of dynamic loading, Hub kernels differ from traditional Python kernel
packages in that they are made to be:

- Portable: a kernel can be loaded from paths outside `PYTHONPATH`.
- Unique: multiple versions of the same kernel can be loaded in the
  same Python process.
- Compatible: kernels must support all recent versions of Python and
  the different PyTorch build configurations (various CUDA versions
  and C++ ABIs). Furthermore, older C library versions must be supported.

`kernel-builder` is a set of tools that can build conforming kernels. It
takes care of:

- Building kernels for all supported PyTorch configurations (C++98/11 and
  different CUDA versions).
- Compatibility with old glibc and libstdc++ versions, so that kernels also
  work on older Linux distributions.
- Registering Torch ops, such that multiple versions the same kernel can be
  loaded without namespace conflicts.

`kernel-builder` builds are configured through a `build.toml` file.
`build.toml` is a simple format that does not require intricate knowledge
of CMake or setuptools.

This page describes the directory layout of a kernel-builder project, the
format of the `build.toml` file, and some additional Python glue that
`kernel-builder` provides. We will use a [simple ReLU kernel](https://github.com/huggingface/kernels/tree/main/examples/kernels/relu)
as the running example. After reading this page, you may also want to have
a look at the more realistic [ReLU kernel with backprop and `torch.compile`](https://github.com/huggingface/kernels/tree/main/examples/kernels/relu-backprop-compile)
support.

> [!TIP]
> We maintain a set of conforming kernels in the
> [kernels-community repository](https://github.com/huggingface/kernels-community).
> We try to keep these kernels synced with upstream as much as possible.

## Setting up environment

### Quick install

The fastest way to get started is to run the install script. This
installs [Determinate Nix](https://docs.determinate.systems/determinate-nix/)
and `kernel-builder` in a single command:

```bash
curl -fsSL https://raw.githubusercontent.com/huggingface/kernels/main/install.sh | bash
```

This will:

1. Install Determinate Nix (if not already installed).
2. Configure the Hugging Face binary cache (to avoid building dependencies from
   source).
3. Install `kernel-builder` via `nix profile install`.

To update `kernel-builder` later:

```bash
nix profile upgrade --all
```

For a step-by-step breakdown of what the script does, see
[Using the kernel builder with Nix](nix.md).

### Cloud environment

In the [`terraform`](https://github.com/huggingface/kernels/tree/main/terraform) directory, we provide an
example of programatically spinning up an EC2 instance that is ready
with everything needed for you to start developing and building
kernels.

If you use a different provider, the Terraform bridges should be
similar and straightforward to modify.

## Starting a new kernel

The easiest way to start a new kernel is by using the `init` subcommand
of `kernel-builder`. This creates a minimal, compilable kernel:

```bash
$ kernel-builder init --name myorg/mykernel
Initialized `myorg/mykernel` at /home/daniel/git/kernels/examples/kernels/mykernel
```

This creates a kernel named `mykernel` in the directory `mykernel`. The
kernel is configured to upload to the `myorg/mykernel` Hub
repository when an upload command is used.

By default, the `init` subcommand creates a CUDA kernel. You can specify
another backend with the `--backends` option:

```bash
$ kernel-builder init --name myorg/mykernel --backends xpu
```

You can also make a multi-backend kernel by adding all the backends
that you would like to support as arguments to `--backends`:

```bash
$ kernel-builder init --name myorg/mykernel --backends cuda xpu
Initialized `myorg/mykernel` at /home/daniel/git/kernels/examples/kernels/mykernel
```

Finally, if you want to create a kernel for all supported backends, you
can use `--backends all`.

## Kernel project layout

Kernel projects follow this general directory layout:

```text
mykernel
├── benchmarks
│   └── benchmark.py
├── build.toml
├── CARD.md
├── example.py
├── flake.nix
├── mykernel_cuda
│   └── mykernel.cu
├── tests
│   ├── __init__.py
│   └── test_mykernel.py
└── torch-ext
├── mykernel
│   └── __init__.py
├── torch_binding.cpp
└── torch_binding.h
```

In this example we can find:

- The build configuration in `build.toml`.
- One or more top-level directories containing kernels (`mykernel_cuda`).
- The `torch-ext` directory, which contains:
  - `torch_binding.h`: contains declarations for kernel entry points
    (from `kernel_a` and `kernel_b`).
  - `torch_binding.cpp`: registers the entry points as Torch ops.
  - `torch_ext/mykernel`: contains any Python wrapping the kernel needs. At the
    bare minimum, it should contain an `__init__.py` file.
- Kernel tests in the directory `tests`.
- Benchmarks in the directory `benchmarks`.
- A kernel card template in `CARD.md`. This placeholders in the card are filled
  during the kernel build.
- The Nix flake configuration in `flake.nix`.
- An example script that uses the kernel in `example.py`.

## `build.toml`

`build.toml` tells `kernel-builder` what to build and how. It looks as
follows for the `mykernel` kernel:

```toml
[general]
backends = [
  "cuda",
]
name = "mykernel"
version = 1

[general.hub]
repo-id = "myorg/mykernel"

[torch]
src = [
  "torch-ext/torch_binding.cpp",
  "torch-ext/torch_binding.h",
]

[kernel.mykernel]
backend = "cuda"
depends = ["torch"]
src = ["mykernel_cuda/mykernel.cu"]
# If the kernel is only supported on specific capabilities, set the
# cuda-capabilities option:
#
# cuda-capabilities = [ "9.0", "10.0", "12.0" ]
```

The following sections enumerate all supported options for `build.toml`.

### `general`

- `name` (required): the name of the kernel. The Python code for a Torch
  extension must be stored in `torch-ext/<name>`.
- `version` (int): the major version of the kernel.
  The version is written to the kernel's `metadata.json` and is used
  by the `kernels upload` command to upload the kernel to a version
  branch named `v<version>`.
- `backends` (required): a list of supported backends. Must be one or
  more of `cpu`, `cuda`, `metal`, `rocm`, or `xpu`.
- `upstream`: Git-compatible URL (passable to `git clone`) of the original
  upstream repository where the kernel source code comes from.
- `source`: Git-compatible URL (passable to `git clone`) of the kernel-builder
  formatted source repository. This repository must contain a `build.toml` and
  `flake.nix` so that it can be pulled and built with the kernel builder.
- `python-depends` (**experimental**): a list of additional Python dependencies
  that the kernel requires. The only supported dependencies are `einops`
  and `nvidia-cutlass-dsl`.

### `general.hub`

- `repo-id`: the Hub repository to upload the kernel to when the `upload` or
  `build-and-upload` subcommands of `kernel-builder` are used.

### `general.cuda`

- `maxver`: the maximum CUDA toolkit version (inclusive). This option
  _must not_ be set under normal circumstances, since it can exclude Torch
  build variants that are [required for compliant kernels](../kernel-requirements.md).
  This option is provided for kernels that cause compiler errors on
  newer CUDA toolkit versions.
- `minver`: the minimum required CUDA toolkit version. This option
  _must not_ be set under normal circumstances, since it can exclude Torch
  build variants that are [required for compliant kernels](../kernel-requirements.md).
  This option is provided for kernels that require functionality only
  provided by newer CUDA toolkits.

### Framework sections

The framework section specifies framework-specific settings. The name of
the section depends on the framework that is used. The currently supported
frameworks are:

- AOT-compiled Torch kernel (`torch`).
- AOT-compiled TVM-FFI kernel (`tvm-ffi`).
- JIT-compiled or not-compiled Torch kernel (`torch-noarch`, experimental).

### `torch`

This framework section is used for AOT-compiled Torch kernels, and has the
following options:

- `src` (required): a list of source files and headers.
- `pyext` (optional): the list of extensions for Python files. Default:
  `["py", "pyi"]`.
- `include` (optional): include directories relative to the project root.
  Default: `[]`.
- `maxver` (optional): only build for this Torch version and earlier. Use cautiously, since this option produces
  non-compliant kernels if the version range does not correspond to the [required variants](build-variants.md).
- `minver` (optional): only build for this Torch version and later. Use cautiously, since this option produces
  non-compliant kernels if the version range does not correspond to the [required variants](build-variants.md).
- `stable-abi` (**experimental**): when set to a Torch version (e.g.
  `"2.11"`), the kernel is built using the Torch stable ABI. This
  requires that the kernel itself only use
  [stable ABI headers](https://docs.pytorch.org/docs/2.12/notes/libtorch_stable_abi.html).
  For an example, see the [`relu-torch-stable-abi`](https://github.com/huggingface/kernels/tree/main/examples/kernels/relu-torch-stable-abi)
  example kernel.

### `tvm-ffi`

This framework section is used for AOT-compiled TVM-FFI kernels.

- `src` (required): a list of source files and headers.
- `pyext` (optional): the list of extensions for Python files. Default:
  `["py", "pyi"]`.
- `include` (optional): include directories relative to the project root.
  Default: `[]`.

### `torch-noarch`

The `torch-noarch` section is used for JIT-compiled kernels or kernels that
do not require any ahead-of-time compilation (e.g. a kernel that packages plain PyTorch
layers).

Normally, it is expected that this type of kernel runs on all CUDA capabilities
or ROCm architectures. However, for kernels that support only a limited range
of archs, the `cuda-capabilites` and `rocm-archs` options can be used to specify
the supported archs. These are then exported to `metadata.json` for consumption
by e.g. the Hugging Face Hub.

- `pyext` (optional): the list of extensions for Python files. Default:
  `["py", "pyi"]`.
- `cuda-capabilities` (optional): a list of CUDA compute capabilities the
  kernel supports (e.g. `["9.0", "10.0"]`).
- `rocm-archs` (optional): a list of ROCm architectures the kernel supports
  (e.g. `["gfx942"]`).

### `kernel.<name>`

Specification of a kernel with the name `<name>`. Multiple `kernel.<name>`
sections can be defined in the same `build.toml`.
See for example [`kernels-community/quantization`](https://huggingface.co/kernels-community/quantization/)
for an example with multiple kernel sections.

The following options can be set for a kernel:

- `backend` (required): the compute backend of the kernel. The currently
  supported backends are `cpu`, `cuda`, `metal`, `rocm`, and `xpu`.
  **The `cpu` backend is currently experimental and might still change.**
- `depends` (required): a list of dependencies. The supported dependencies
  are listed in [`deps.nix`](https://github.com/huggingface/kernels/blob/main/builder/lib/deps.nix).
- `src` (required): a list of source files and headers.
- `include` (optional): include directories relative to the project root.
  Default: `[]`.

Besides these shared options, the following backend-specific options
are available:

#### cuda

- `cuda-capabilities` (optional): a list of CUDA capabilities that the
  kernel should be compiled for. When absent, the kernel will be built
  using all capabilities that the builder supports. The effective
  capabilities are the intersection of this list and the capabilities
  supported by the CUDA compiler. It is recommended to leave this option
  unspecified **unless** a kernel requires specific capabilities.
- `cuda-flags` (optional): additional flags to be passed to `nvcc`.
  **Warning**: this option should only be used in exceptional circumstances.
  Custom compile flags can interfere with the build process or break
  compatibility requirements.

#### rocm

- `rocm-archs`: a list of ROCm architectures that the kernel should be
  compiled for.

#### xpu

- `sycl-flags`: a list of additional flags to be passed to the SYCL
  compiler.

### cpu

- `cxx-flags`: a list of additional flags to be passed to the C++
  compiler.

## Torch bindings

### Defining bindings

Torch bindings are defined in C++, kernels commonly use two files:

- `torch_binding.h` containing function declarations.
- `torch_binding.cpp` registering the functions as Torch ops.

For instance, the `mykernel` kernel discussed above has the following
declaration in `torch_binding.h`:

```cpp
#pragma once

#include <torch/torch.h>

void mykernel(torch::Tensor &out, torch::Tensor const &input);
```

This function is then registered as a Torch op in `torch_binding.cpp`:

```cpp
#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("mykernel(Tensor! out, Tensor input) -> ()");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("mykernel", torch::kCUDA, &mykernel);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
```

This snippet uses macros from `registration.h` to register the function.
`registration.h` is generated by `kernel-builder` itself. A function
is registered through the `def`/`ops` methods. `ops` specifies the
function signature following the [function schema](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func).
`impl` associates the function name with the C/C++ function and
the applicable device.

## Using kernel functions from Python

The bindings are typically wrapped in Python code in `torch_ext/<name>`.
The native code is exposed under the `torch.ops` namespace. However,
we add some unique material to the name of the extension to ensure that
different versions of the same extension can be loaded at the same time.
As a result, the extension is registered as
`torch.ops.<name>_<unique_material>`.

To deal with this uniqueness, `kernel_builder` generates a Python module
named `_ops` that contains an alias for the name. This can be used to
refer to the correct `torch.ops` module. For example:

```python
from typing import Optional

import torch

from ._ops import ops


def mykernel(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x)
    ops.mykernel(out, x)
    return out
```

## Registering Torch operators

You may want to register Torch ops from your kernel's Python code or
register fake ops for `torch.compile` support. It is important to register
such ops in the namespace that kernel-builder makes for your kernel
build. This is required for compliant kernels to ensure that multiple
versions of the same kernel can be loaded at the same time without
namespace conflicts.

You can use the `add_op_namespace_prefix` to prefix an op name with the
correct prefix. So for instance, replace

```python
@torch.library.register_fake("relu::relu_fwd")
def relu_fwd_fake(input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)
```

by

```python
from ._ops import add_op_namespace_prefix

@torch.library.register_fake(add_op_namespace_prefix("relu_fwd"))
def relu_fwd_fake(input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)
```

As mentioned in the above, the `_ops` module is generated by kernel-builder.

kernel-builder uses a hook to reject incorrect usage of Torch op registration
functions. However, it can only catch direct use of certain `torch.library`
decorators. For instance, the hook would not reject the following decorator,
so it should be seen as a last-resort check if human review failed:

```python
@some_indirection_for_register_fake("relu::relu_fwd")
def relu_fwd_fake(input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)
```

## Kernel tests

Kernel tests are stored in the `tests` directory. Since running all
kernel tests in CI may be prohibitively expensive, the `pyproject.toml`
generated by the builder adds support for the special `kernels_ci`
PyTest marker that can be used as follows:

```python
import pytest

@pytest.mark.kernels_ci
def test_mykernel():
  ...
```

We recommend that you to pick tests that together would catch most error
cases while running within 60 seconds.

You can run the tests (e.g. in CI) using:

```bash
$ nix run .#ci-test
```

If the kernel supports multiple backends, it will run the test for the
first supported backend that was found, obeying the following order: CUDA,
ROCm, XPU, Metal, CPU. If you would like to the tests for a specific build
variant, you can use `nix run .#ciTests.<variant>`. For instance:

```bash
$ nix run .#ciTests.torch210-cxx11-cpu-x86_64-linux
```

When running the tests on a non-NixOS systems, make sure that
[the CUDA driver library can be found](https://danieldk.eu/Software/Nix/Nix-CUDA-on-non-NixOS-systems#solutions).

## Kernel docs

We provide a utility to generate a system card for a given kernel, utilizing
information from its `build.toml` and metadata. This system card provides a
reasonable starting point and is meant to be edited afterward by the kernel
developer.

The template card is generated as a part of `kernel-builder init`
command and is serialized in the root directory of the kernel.

The card will be filled automatically by the builder when using the
`build-and-upload` or `build-and-copy` command. It will be serialized
to the `build` sub-directory inside the main kernel directory. It
will be uploaded as `README.md` to the Hub.
