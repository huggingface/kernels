# Using the kernel builder with Nix

The kernel builder uses Nix for building kernels. You can build or
run the kernels directly if you have Nix installed on your system.
We recommend installing Nix in the following way:

- Linux: use the [official Nix installer](https://nixos.org/download/).
- macOS: use the [Determinate Nix installer](https://docs.determinate.systems/determinate-nix/).
  In addition, Xcode 16.x is currently required to build kernels.

## Getting started

The easiest way get all the Nix functionality is by putting a
`flake.nix` in your kernel repository. To do so, copy
[`examples/relu/flake.nix`](../examples/relu/flake.nix) into the
same directory as your `build.toml` file. Then run `nix flake update`.
This generates a `flake.lock` file that pins the kernel builder
and _all_ its transitive dependencies. Commit both `flake.nix`
and `flake.lock` to your repository, this will ensure that kernel
builds are reproducible.

Since the kernel builder depends on many packages (e.g. every supported
PyTorch version), it is recommended to enable the huggingface cache
to avoid expensive rebuilds.

To use the cache, you can either install cachix and configure it:

```bash
# Install cachix and configure the cache
cachix use huggingface
```

Or run it once without installing cachix permanently:

```bash
# Use cachix without installing it
nix run nixpkgs#cachix -- use huggingface
```

The kernel builder also provides Nix development shells with all Torch
and CUDA/ROCm dependencies needed to develop kernels (see below). If
you want to test your kernels inside a Nix development shell and you
are not using NixOS, [make sure that the CUDA driver is visible](https://danieldk.eu/Nix-CUDA-on-non-NixOS-systems#make-runopengl-driverlib-and-symlink-the-driver-library) to Torch.

## Building kernels with Nix

A kernel that has a `flake.nix` file can be built with the `build-and-copy`
command. For example:

```bash
cd examples/relu
nix run .#build-and-copy -L
```

The compiled kernel will then be in the local `build/` directory.

## Shell for local development

`kernel-builder` provides shells for developing kernels. In such a shell,
all required dependencies are available, as well as `build2cmake` for generating
project files. For example:

```bash
$ nix develop
$ build2cmake generate-torch build.toml
$ cmake -B build-ext
$ cmake --build build-ext
```

If you want to test the kernel as a Python package, you can do so.
`nix develop` will automatically create a virtual environment in the
`.venv` and activate it. You can install the kernel as a regular
Python package in this virtual environment:

```bash
$ nix develop
$ build2cmake generate-torch build.toml
$ pip install --no-build-isolation -e .
```

Development shells are available for every build configuration. For
instance, you can get a Torch 2.7 development shell for ROCm extensions
using:

```bash
$ rm -rf .venv  # Remove existing venv if any.
$ nix develop .#devShells.torch29-cxx11-rocm64-x86_64-linux
```

## Shell for testing a kernel

You can also start a development shell. This will give you a Python interpreter
with the kernel in Python's search path. This makes it more convenient to run
tests:

```bash
cd examples/relu
nix develop -L .#test
python -m pytest tests
```

## Adding test dependencies to development shells

You can add test dependencies to a development or testing shell. Adapt
the kernel's `flake.nix` to use the `pythonCheckInputs` option:

```nix
{
  description = "Flake for my kernel";

  inputs = {
    builder.url = "github:huggingface/kernels";
  };

  outputs =
    {
      self,
      builder,
    }:
    builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;

      # The einops and numpy test dependencies are added here:
      pythonCheckInputs = pkgs: with pkgs; [ einops numpy ];
    };
}
```

The available packages can be found on [search.nixos.org](https://search.nixos.org/packages?channel=25.05&query=python312Packages).

Keep in mind that these additional dependencies will only be available to
the Nix shells, not the final kernel uploaded to the Hub.

## Skipping the `get_kernel` check

`kernel-builder` verifies that a kernel can be
imported with the [`kernels`](https://github.com/huggingface/kernels)
package. This check can be disabled by passing `doGetKernelCheck = false`
to `genKernelFlakeOutputs`. **Warning:** it is strongly recommended to keep
this check enabled, as it is one of the checks that validates that a kernel
is compliant. This option is primarily intended for kernels with
`triton.autotune` decorators, which can fail because there is no GPU available
in the build sandbox.
