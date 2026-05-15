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
    # Base package set.
    (import ./components.nix)
    # Overrides (adding dependencies, etc.)
    (import ./overrides.nix)

    # Unwrapped gcc (gcc13-unwrapped, etc.)
    (import ./gcc-unwrapped.nix)

    # stdenvs (gcc13Stdenv, etc.)
    (import ./stdenv.nix { inherit pkgs; })

    # Use the gcc14 stdenv by default.
    (final: prev: { stdenv = final.gcc14Stdenv; })

    # Create a CUDA stdenv that uses a gcc that is compatible with the
    # default CUDA version of the package set.
    (import ./cuda-backend-stdenv.nix { inherit pkgs; })

  ];
in
lib.makeScope newScope (lib.extends composed fixedPoint)
