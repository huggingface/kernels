# Nix Builder design

## Introduction

kernel-builder uses a Nix-based builder that orchestrates the build. The Nix
builder provides:

- Reproducible evaluation. The same Nix builder version will always produce
  the same derivations (build recipes).
- Largely reproducible builds by using a build sandbox that only has the
  dependencies specified in a derivation.
- Seamless creation of different build environments (e.g. different Torch
  and CUDA combinations).

## Kernel build steps

A kernel derivation builds a kernel in the following steps:

1. Generate CMake files for the kernel using
   `kernel-builder create-pyproject`.
2. Generate Ninja build files using CMake.
3. Build the kernel using Ninja.
4. Perform various checks on the compiled kernel, such as:
   - Verify that the kernel only uses ABI3/`manylinux_2_28` symbols.
   - Verify that the kernel can be loaded by the `kernels` Python package.
5. Strip runpaths (ELF-embedded library directories) from kernel binaries
   to make the kernel distribution-independent.

## manylinux_2_28 compatibility

To achieve `manylinux_2_28` compatibility, kernels are built using a
toolchain similar to the `manylinux_2_28` Docker images. This toolchain
is based on the gcc toolsets from AlmaLinux 8. `manylinux_2_28` [uses
AlmaLinux 8 as its base](https://github.com/pypa/manylinux#manylinux_2_28-almalinux-8-based),
so we have to compile against the same glibc/libstdc++ versions to
ensure compatibility.

We repackage the AlmaLinux 8 toolsets and libstdc++ as Nix derivations (see
the `nix-builder/packages/manylinux_2_28` source directory). Then we merge
various toolset packages to an unwrapped gcc that resembles unwrapped gcc in
nixpkgs. Finally, we wrap binutils and gcc to combine them into a stdenv.

The stdenv does not reuse glibc from AlmaLinux, since its dynamic loader has
hardcoded FHS paths (`/lib64` etc.) that are not valid in Nix. Using this
dynamic loader results in linking errors, since the paths in the dynamic
loader are used as a last resort (to link glibc libraries). So, instead we
build our own glibc 2.28 package and use that.

## The package set pattern

We repackage various existing package sets as Nix derivations. For instance,
this is done for ROCm, XPU, and manylinux_2_28 packages. These package sets
all follow the same pattern:

```nix
{
  lib,
  callPackage,
  newScope,
  pkgs,
}:

{
  packageMetadata,
}:

let
  inherit (lib.fixedPoints) extends composeManyExtensions;

  fixedPoint = final: {
    inherit lib;
  };
  composed = lib.composeManyExtensions [
    # Base package set.
    (import ./components.nix { inherit packageMetadata; })

    # Package-specific overrides.
    (import ./overrides.nix)

    # Additional overlays that extend the package set.
    (import ./some-overlay.nix)
  ];
in
lib.makeScope newScope (lib.extends composed fixedPoint)
```

We use a fixed point to build up the package set as a list of
[overlays](https://nixos.org/manual/nixpkgs/stable/#sec-overlays-definition).
This has various benefits. For instance, it allows us to refine the
package set incrementally and we can refer to the final versions of
packages in intermediate overlays.

The package sets all use a similar list of overlays:

- An initial overlay (`components.nix`) that applies a generic builder
  to the package set metadata. The metadata typically comes from a Yum/DNF
  repository that contains RPM packages.The generic builder will extract the
  RPMs and move binaries, libraries, and headers to the right location. This
  results in a set of Nix derivations that may or may not build.
- The next overlay (`overrides.nix`) fixes up derivations created by the
  generic builder that do not build. Fixing the derivations typically consists
  of adding missing dependencies and changing embedded FHS paths to Nix store
  paths.
- Additional overlays with derivations that combine outputs from previous
  overlays. One typical example are derivations that construct a full compiler
  toolchain.
