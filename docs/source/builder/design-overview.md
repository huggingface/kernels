# Architecture overview

The Kernel Builder is a Nix flake that combines two components:

![Kernel Builder architecture](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/kernels/kernel_design_overview.jpg)

- `kernel-builder/`: a Rust-based CLI
  that parses and validates `build.toml`, scaffolds new kernel projects, and drives the
  build. It does not compile kernels itself: every build subcommand executes
  `nix build`/`nix run`/`nix develop` against the kernel project's `flake.nix`.
- `nix-builder`: the Nix builder drives the build itself. It generates a build matrix using the supported build configurations
  (see `nix-builder/versions.nix`) and the kernel's `build.toml`. The build matrix is the cartesian product of
  the backend (one or more of CPU/CUDA/Metal/ROCm/XPU), backend versions, and framework versions.
  Nix builder uses `kernel-builder` to generate the CMake files that drive the build. To do so,
  `kernel-builder` itself is packaged in `nix-builder/pkgs`.

A kernel author uses the Nix builder through the `lib.genKernelFlakeOutputs` that is exposed through the top-level `flake.nix`. This generates the Nix flake outputs for building and developing kernels, such as `bundle` for bundle builds and `devShells` for development shells.

For a deeper look at the design of the Nix builder, see
[Nix Builder design](./design-nix-builder).
