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

    # Unwrapped gcc (gcc13-unwrapped, gcc14-unwrapped, etc.)
    (import ./gcc-unwrapped.nix)

    # stdenvs (gcc13Stdenv, gcc14Stdenv, etc.)
    (import ./stdenv.nix { inherit pkgs; })

    # Use the gcc14 stdenv by default.
    (final: prev: { stdenv = final.gcc14Stdenv; })

    # Create a CUDA stdenv that uses a gcc that is compatible with the
    # CUDA version set as the default in nixpkgs.
    (import ./cuda-backend-stdenv.nix { inherit pkgs; })

  ];
in
lib.makeScope newScope (lib.extends composed fixedPoint)
