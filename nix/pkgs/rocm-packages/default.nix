{
  lib,
  callPackage,
  newScope,
}:

{
  packageMetadata,
}:

let
  fixedPoint = final: { inherit callPackage lib packageMetadata; };
  composed = lib.composeManyExtensions [
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
      inherit (prev.callPackage ../aotriton { })
        aotriton_0_10
        aotriton_0_11
        aotriton_0_11_1
        ;
    })
  ];
in
lib.makeScope newScope (lib.extends composed fixedPoint)
