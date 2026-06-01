# Design overview

Kernel Builder is structured as **two cooperating layers**, glued together by the
top-level `flake.nix`:

![Kernel Builder architecture](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/kernels/kernel_design_overview.jpg)

- **`kernel-builder/` — the Rust CLI (user-facing entry point).** A clap-based binary
  that parses and validates `build.toml`, scaffolds new kernel projects, and drives the
  build. It does not compile kernels itself: every build subcommand shells out to
  `nix build`/`nix run`/`nix develop` against the kernel project's `flake.nix`. Rust is
  the orchestration/UX layer, not the build engine.
- **`nix-builder/` — the Nix build infrastructure (the actual builder).** The Nix
  functions in `lib/` turn a kernel project into a matrix of backend (CUDA / ROCm / XPU /
  Metal) × framework × framework-version × Python-version variants and drive the
  compilation in a Nix sandbox. The `pkgs/` derivations provide everything the build
  needs — including the Rust CLI itself, packaged with `rustPlatform.buildRustPackage`.

The two layers stay in sync through shared sources of truth (e.g.
`cuda_supported_archs.json` is read by both), and the top-level flake exposes
`lib.genKernelFlakeOutputs`, which a kernel author's flake calls to get their full build
matrix.

For a deeper look at how the Nix layer builds a kernel — the build steps,
`manylinux_2_28` compatibility, and the package-set pattern — see the
[Nix Builder design](./design-nix-builder).
