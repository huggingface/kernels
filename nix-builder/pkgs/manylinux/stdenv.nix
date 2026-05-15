{ pkgs }:

final: prev:

let
  nixpkgs_20191230 = import (pkgs.fetchFromGitHub {
    owner = "NixOS";
    repo = "nixpkgs";
    rev = "a9eb3eed170fa916e0a8364e5227ee661af76fde";
    hash = "sha256-1ycrr9HMrGA3ZDM8qmKcZICBupE5UShnIIhPRWdvAzA=";
  }) { inherit (pkgs.stdenv.hostPlatform) system; };

  # Fails to build with current gcc versions, so build with an
  # old gcc from around the time glibc 2.28 was released.
  glibc_2_28 = nixpkgs_20191230.callPackage ./glibc_2_28 { };

  mkBinutilsWrapped =
    { binutils, glibc }:
    pkgs.wrapBintoolsWith {
      bintools = binutils;
      libc = glibc;
    };
  mkGccWrapped =
    {
      binutils,
      coreutils,
      gcc-unwrapped,
      glibc,
      wrapCCWith,
    }:
    wrapCCWith {
      bintools = mkBinutilsWrapped {
        inherit binutils glibc;
      };
      coreutils = coreutils;
      cc = gcc-unwrapped;
      gccForLibs = gcc-unwrapped;
      libc = glibc;
      isGNU = true;
      useCcForLibs = true;
    };
  mkStdenv =
    {
      binutils,
      coreutils,
      gcc-unwrapped,
      glibc,
      overrideCC,
      stdenv,
      wrapCCWith,
    }:
    overrideCC stdenv (mkGccWrapped {
      inherit
        binutils
        coreutils
        gcc-unwrapped
        glibc
        wrapCCWith
        ;
    });
in

builtins.listToAttrs (
  map
    (version: {
      name = "gcc${version}Stdenv";
      value = pkgs.overrideCC pkgs.stdenv (mkGccWrapped {
        inherit (pkgs) wrapCCWith;
        binutils = final."gcc-toolset-${version}-binutils";
        coreutils = final.coreutils;
        gcc-unwrapped = final."gcc${version}-unwrapped";
        # We are using our own glibc 2.28, since the Red Hat version has
        # the wrong glibc paths baked into the dynamic loader.
        glibc = glibc_2_28;
      });
    })
    [
      "13"
      "14"
    ]
)
