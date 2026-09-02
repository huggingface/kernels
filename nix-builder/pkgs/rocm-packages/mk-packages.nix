{
  lib,
  callPackage,
  newScope,
}:

manifest:

let
  inherit (lib.fixedPoints) extends composeManyExtensions;

  fixedPoint = final: {
    inherit lib manifest;
  };
  composed = composeManyExtensions [
    # Hooks
    (import ./hooks.nix)
    (callPackage ./patchelf.nix { })
    # Base package set.
    (import ./components.nix)
    # Overrides (adding dependencies, etc.)
    (import ./overrides.nix)
    # Compiler toolchain.
    (callPackage ./llvm.nix { })
    # ROCm compilation toolchain.
    (final: prev: {
      clr = final.callPackage ./clr.nix {
        inherit (final.llvm) clang;
      };
    })
    # Add aotriton
    (final: prev: {
      inherit (final.callPackage ../aotriton { })
        aotriton_0_11_1
        aotriton_0_11_2
        aotriton_0_12
        aotriton_0_13
        ;
    })
    # Remove once the old package set is gone.
    (final: prev: {
      theRock = true;
      version = final.amdrocm.version;
    })
  ];
in
lib.makeScope newScope (extends composed fixedPoint)
