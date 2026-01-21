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
`kernel-builder` provides. We will use a [simple ReLU kernel](../examples/relu)
as the running example. After reading this page, you may also want to have
a look at the more realistic [ReLU kernel with backprop and `torch.compile`](../examples/relu-backprop-compile)
support.

## Kernel project layout

Kernel projects follow this general directory layout:

```text
relu
├── build.toml
├── relu_kernel
│   └── relu.cu
└── torch-ext
    └── torch_binding.cpp
    └── torch_binding.h
    └── relu
        └── __init__.py
```

In this example we can find:

- The build configuration in `build.toml`.
- One or more top-level directories containing kernels (`relu_kernel`).
- The `torch-ext` directory, which contains:
  - `torch_binding.h`: contains declarations for kernel entry points
    (from `kernel_a` and `kernel_b`).
  - `torch_binding.cpp`: registers the entry points as Torch ops.
  - `torch_ext/relu`: contains any Python wrapping the kernel needs. At the
    bare minimum, it should contain an `__init__.py` file.

## `build.toml`

`build.toml` tells `kernel-builder` what to build and how. It looks as
follows for the `relu` kernel:

```toml
[general]
name = "relu"

[torch]
src = [
  "torch-ext/torch_binding.cpp",
  "torch-ext/torch_binding.h"
]

[kernel.relu]
backend = "cuda"
src = [
  "relu_kernel/relu.cu",
]
depends = [ "torch" ]
# If the kernel is only supported on specific capabilities, set the
# cuda-capabilities option:
#
# cuda-capabilities = [ "9.0", "10.0", "12.0" ]
```

### `general`

- `name` (required): the name of the kernel. The Python code for a Torch
  extension must be stored in `torch-ext/<name>`.
- `version` (int, **experimental**): the major version of the kernel.
  The version is written to the kernel's `metadata.json` and is used
  by the `kernels upload` command to upload the kernel to a version
  branch named `v<version>`.
- `backends` (required): a list of supported backends. Must be one or
  more of `cpu`, `cuda`, `metal`, `rocm`, or `xpu`.
- `python-depends` (**experimental**): a list of additional Python dependencies
  that the kernel requires. The only supported dependencies are `einops`
  and `nvidia-cutlass-dsl`.

### `general.cuda`

- `maxver`: the maximum CUDA toolkit version (inclusive). This option
  _must not_ be set under normal circumstances, since it can exclude Torch
  build variants that are [required for compliant kernels](https://github.com/huggingface/kernels/blob/main/docs/kernel-requirements.md).
  This option is provided for kernels that cause compiler errors on
  newer CUDA toolkit versions.
- `minver`: the minimum required CUDA toolkit version. This option
  _must not_ be set under normal circumstances, since it can exclude Torch
  build variants that are [required for compliant kernels](https://github.com/huggingface/kernels/blob/main/docs/kernel-requirements.md).
  This option is provided for kernels that require functionality only
  provided by newer CUDA toolkits.

### `torch`

This section describes the Torch extension. In the future, there may be
similar sections for other frameworks. This section has the following
options:

- `src` (required): a list of source files and headers.
- `pyext` (optional): the list of extensions for Python files. Default:
  `["py", "pyi"]`.
- `include` (optional): include directories relative to the project root.
  Default: `[]`.
- `maxver` (optional): only build for this Torch version and earlier. Use cautiously, since this option produces
  non-compliant kernels if the version range does not correspond to the [required variants](build-variants.md).
- `minver` (optional): only build for this Torch version and later. Use cautiously, since this option produces
  non-compliant kernels if the version range does not correspond to the [required variants](build-variants.md).

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
  are listed in [`deps.nix`](../lib/deps.nix).
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
- `cuda_flags` (optional): additional flags to be passed to `nvcc`.
  **Warning**: this option should only be used in exceptional circumstances.
  Custom compile flags can interfere with the build process or break
  compatibility requirements.

#### rocm

- `rocm-archs`: a list of ROCm architectures that the kernel should be
  compiled for.

#### xpu

- `sycl_flags`: a list of additional flags to be passed to the SYCL
  compiler.

## Torch bindings

### Defining bindings

Torch bindings are defined in C++, kernels commonly use two files:

- `torch_binding.h` containing function declarations.
- `torch_binding.cpp` registering the functions as Torch ops.

For instance, the `relu` kernel has the following declaration in
`torch_binding.h`:

```cpp
#pragma once

#include <torch/torch.h>

void relu(torch::Tensor &out, torch::Tensor const &input);
```

This is a declaration for the actual kernel, which is in `relu_kernel/relu.cu`:

```cpp
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cmath>

__global__ void relu_kernel(float *__restrict__ out,
                            float const *__restrict__ input,
                            const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    auto x = input[token_idx * d + idx];
    out[token_idx * d + idx] = x > 0.0f ? x : 0.0f;
  }
}

void relu(torch::Tensor &out,
          torch::Tensor const &input)
{
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float &&
                  input.scalar_type() == at::ScalarType::Float,
              "relu_kernel only supports float32");

  int d = input.size(-1);
  int64_t num_tokens = input.numel() / d;
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  relu_kernel<<<grid, block, 0, stream>>>(out.data_ptr<float>(),
                                          input.data_ptr<float>(), d);
}
```

This function is then registered as a Torch op in `torch_binding.cpp`:

```cpp
#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("relu(Tensor! out, Tensor input) -> ()");
  ops.impl("relu", torch::kCUDA, &relu);
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

def relu(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x)
    ops.relu(out, x)
    return out
```
