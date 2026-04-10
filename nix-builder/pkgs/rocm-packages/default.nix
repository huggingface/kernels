{
  lib,
  callPackage,
  newScope,
}:

{
  packageMetadata,
}:

let
  inherit (lib.fixedPoints) extends composeManyExtensions;

  fixedPoint = final: {
    inherit lib packageMetadata;
  };
  composed = composeManyExtensions [
    # Hooks
    (import ./hooks.nix)
    # Base package set.
    (import ./components.nix)
    # Overrides (adding dependencies, etc.)
    (import ./overrides.nix)
    # Compiler toolchain.
    (callPackage ./llvm.nix { })
    # Packages that are joins of other packages.
    (callPackage ./joins.nix { })
    # Add aotriton
    (final: prev: {
      inherit (final.callPackage ../aotriton { })
        aotriton_0_11_1
        aotriton_0_11_2
        ;
    })
  ];
in
lib.makeScope newScope (extends composed fixedPoint)
