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

  nixpkgs_20191230 = import (pkgs.fetchFromGitHub {
    owner = "NixOS";
    repo = "nixpkgs";
    rev = "a9eb3eed170fa916e0a8364e5227ee661af76fde";
    hash = "sha256-1ycrr9HMrGA3ZDM8qmKcZICBupE5UShnIIhPRWdvAzA=";
  }) { inherit (pkgs.stdenv.hostPlatform) system; };

  # Fails to build with old gcc, so build with old version.
  glibc_2_28 = nixpkgs_20191230.callPackage ./glibc_2_28 { };

  fixedPoint = final: {
    inherit lib packageMetadata;
  };
  composed = lib.composeManyExtensions [
    # Base package set.
    (import ./components.nix)
    # Overrides (adding dependencies, etc.)
    (import ./overrides.nix)

    (import ./gcc-unwrapped.nix)

    (final: prev: {
      binutils = pkgs.wrapBintoolsWith {
        bintools = final.gcc-toolset-14-binutils;
        libc = glibc_2_28;
      };

      gcc = pkgs.wrapCCWith {
        bintools = final.binutils;
        coreutils = final.coreutils;
        cc = final.gcc14-unwrapped;
        gccForLibs = final.gcc14-unwrapped;
        libc = glibc_2_28;
        isGNU = true;
        useCcForLibs = true;
      };

      stdenv = pkgs.overrideCC pkgs.stdenv final.gcc;
    })
  ];
in
lib.makeScope newScope (lib.extends composed fixedPoint)
