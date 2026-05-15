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
    inherit lib packageMetadata;
  };
  composed = lib.composeManyExtensions [
    # Hooks
    (import ./hooks.nix)
    # Base package set.
    (import ./components.nix)
    # Overrides (adding dependencies, etc.)
    (import ./overrides.nix)
    # Packages that are joins of other packages.
    (final: prev: {
      oneapi-torch-dev = final.callPackage ./oneapi-torch-dev.nix {
        # Wrap the compiler with manylinux 2.28 libraries.
        # TODO: in the future, move dependency resolving to the extension
        #       construction, so that we can pass through the effective
        #       stdenv there.
        inherit (pkgs.manylinux_2_28) stdenv;
      };
    })

    (final: prev: {
      onednn-xpu = final.callPackage ./onednn-xpu.nix { };
    })
    (final: prev: {
      ocloc = final.callPackage ./ocloc.nix { };
    })
    (final: prev: {
      sycl-tla = final.callPackage ./sycl-tla.nix { };
    })
  ];
in
lib.makeScope newScope (lib.extends composed fixedPoint)
