# Using the kernel builder with Nix

## Installation

> [!NOTE]
> The [install script](writing-kernels.md#quick-install) automates
> the Nix and kernel-builder setup described below. Use these manual
> instructions if you prefer step-by-step control.

### Installing Nix

The kernel builder uses Nix for building kernels. You can build or
run the kernels directly if you have Nix installed on your system.
We recommend installing Nix in the following way:

- Linux: use the [official Nix installer](https://nixos.org/download/).
- macOS: use the [Determinate Nix installer](https://docs.determinate.systems/determinate-nix/).
  In addition, Xcode 16.x is currently required to build kernels.

### Using the Hugging Face binary cache

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

### GPU library configuration

The kernel builder also provides Nix development shells with all Torch
and CUDA/ROCm dependencies needed to develop kernels (see below). If
you want to test your kernels inside a Nix development shell and you
are not using NixOS, [make sure that the CUDA driver is visible](https://danieldk.eu/Nix-CUDA-on-non-NixOS-systems#make-runopengl-driverlib-and-symlink-the-driver-library) to Torch.

## Getting started

The easiest way to start a new kernel is using the `kernel-builder init`
subcommand, which is discussed in [Writing Kernels](writing-kernels.md).
The commands discussed in the following sections will also work on
existing kernel sources that have `build.toml`/`flake.nix`.

## Building a kernel

A kernel can be built with the `kernel-builder build-and-copy` command.
For example:

```bash
cd examples/relu
kernel-builder build-and-copy -L
```

The `-L` option prints out build logs in the terminal, which can be handy
for monitoring the build. The compiled kernel will then be in the local
`build/` directory.

## Shell for local development

`kernel-builder` provides shells for developing kernels. In such a shell,
all required dependencies are available, as well as `kernel-builder` for generating
project files. For example:

```bash
$ kernel-builder devshell
# A devshell is opened in which you can run the following commands:
$ kernel-builder create-pyproject
$ cmake -B build-ext
$ cmake --build build-ext
```

If you want to test the kernel as a Python package, you can do so.
`kernel-builder devshell` will automatically create a virtual environment in
the `.venv` and activate it. You can install the kernel as a regular
Python package in this virtual environment:

```bash
$ kernel-builder devshell
$ kernel-builder create-pyproject
$ pip install --no-build-isolation -e .
```

Development shells are available for every build configuration. For
instance, you can get a Torch 2.11 development shell for ROCm kernels
using:

```bash
$ rm -rf .venv  # Remove existing venv if any.
$ kernel-builder devshell --variant torch211-cxx11-rocm71-x86_64-linux
```

You can list the variants that the kernel supports with the `list-variants`
subcommand:

```bash
$ kernel-builder list-variants
torch29-cxx11-cu129-x86_64-linux
torch210-cxx11-cu126-x86_64-linux
torch210-cxx11-cu128-x86_64-linux
torch210-cxx11-cu130-x86_64-linux
torch210-cxx11-rocm70-x86_64-linux
torch210-cxx11-rocm71-x86_64-linux
torch210-cxx11-cpu-x86_64-linux
torch210-cxx11-xpu20253-x86_64-linux
torch211-cxx11-cpu-x86_64-linux
torch211-cxx11-cu126-x86_64-linux
torch211-cxx11-cu128-x86_64-linux
torch211-cxx11-cu130-x86_64-linux
torch211-cxx11-rocm71-x86_64-linux
torch211-cxx11-rocm72-x86_64-linux
torch211-cxx11-xpu20253-x86_64-linux
```

## Shell for testing a kernel

You can also start a test shell. This will give you a Python interpreter
with the kernel in Python's search path. This makes it more convenient to run
tests:

```bash
cd examples/relu
kernel-builder testshell
python -m pytest tests
```

`testshell` also supports the `--variant` option, so you can test a particular
kernel variant.

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
      pythonCheckInputs = pkgs: with pkgs; [ numpy ];
    };
}
```

The available packages can be found on [search.nixos.org](https://search.nixos.org/packages?channel=25.05&query=python312Packages).

Keep in mind that these additional dependencies will only be available to
the Nix shells, not the final kernel uploaded to the Hub.

## Uploading your kernel to the Hub

Finally, when you are ready to make a kernel release, you can build and
upload a kernel to the Hub:

```bash
$ cd mykernel
$ kernel-builder build-and-upload
```

Aside from building and uploading the kernel itself, this will also fill
the card template and upload it as `README.md` to the Hub if the card
template is provided in the source repository as `CARD.md`.

The repository to upload to is determined by the `repo-id` and `version`
fields in `build.toml`. For example, with the following `build.toml`, the
kernel will be uploaded to the repository `kernels-community/flash-attn4`
in the `v1` version branch:

```toml
[general]
# ...
version = 1

[general.hub]
repo-id = "kernels-community/flash-attn4"
```

See [Writing Kernels](writing-kernels.md) for more details on the `build.toml`
format.

## Updating the kernel build toolchain

The kernel's dependencies are fully pinned down in the `flake.lock` that
is shipped with the kernel. We periodically release new versions of the
build toolchain that includes bug fixes and supports newer Torch and compute backend
versions. To update the kernel build toolchain, run `nix flake update`
in the kernel directory:

```bash
❯ nix flake update
• Added input 'kernel-builder':
    'github:huggingface/kernels/8ad8a5094f1b3c425f70900699ed690d65d878c3?narHash=sha256-m8tBntCIlH/rY4BcIv5X5%2BdtgSS1yQi883Co%2Bj5cudI%3D' (2026-04-09)
• Added input 'kernel-builder/flake-compat':
    'github:edolstra/flake-compat/5edf11c44bc78a0d334f6334cdaf7d60d732daab?narHash=sha256-vNpUSpF5Nuw8xvDLj2KCwwksIbjua2LZCqhV1LNRDns%3D' (2025-12-29)
• Added input 'kernel-builder/flake-utils':
    'github:numtide/flake-utils/11707dc2f618dd54ca8739b309ec4fc024de578b?narHash=sha256-l0KFg5HjrsfsO/JpG%2Br7fRrqm12kzFHyUHqHCVpMMbI%3D' (2024-11-13)
• Added input 'kernel-builder/flake-utils/systems':
    'github:nix-systems/default/da67096a3b9bf56a91d16901293e51ba5b49a27e?narHash=sha256-Vy1rq5AaRuLzOxct8nz4T6wlgyUR7zLU309k9mBC768%3D' (2023-04-09)
• Added input 'kernel-builder/nixpkgs':
    'github:NixOS/nixpkgs/2f4fd5e1abf9bac8c1d22750c701a7a5e6b524c6?narHash=sha256-Mh6bLcYAcENBAZk3RoMPMFCGGMZmfaGMERE4siZOgP4%3D' (2026-03-31)
• Added input 'kernel-builder/rust-overlay':
    'github:oxalica/rust-overlay/962a0934d0e32f42d1b5e49186f9595f9b178d2d?narHash=sha256-JMdDYn0F%2BswYBILlpCeHDbCSyzqkeSGNxZ/Q5J584jM%3D' (2026-03-31)
• Added input 'kernel-builder/rust-overlay/nixpkgs':
    follows 'kernel-builder/nixpkgs'
```

## Skipping the `get_kernel` check

`kernel-builder` verifies that a kernel can be
imported with the [`kernels`](https://github.com/huggingface/kernels)
package. This check can be disabled by passing `doGetKernelCheck = false`
to `genKernelFlakeOutputs`. **Warning:** it is strongly recommended to keep
this check enabled, as it is one of the checks that validates that a kernel
is compliant. This option is primarily intended for kernels with
`triton.autotune` decorators, which can fail because there is no GPU available
in the build sandbox.
