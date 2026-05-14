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

    (final: prev: { stdenv = final.gcc14Stdenv; })
  ];
in
lib.makeScope newScope (lib.extends composed fixedPoint)
