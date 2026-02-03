{
  config,
  cudaSupport ? config.cudaSupport,
  fetchFromGitHub,
  overrideCC,
  wrapBintoolsWith,
  wrapCCWith,
  gcc13Stdenv,
  stdenv,
  bintools-unwrapped,
  cudaPackages,
  libgcc,
}:

let
  nixpkgs_20191230 = import (fetchFromGitHub {
    owner = "NixOS";
    repo = "nixpkgs";
    rev = "a9eb3eed170fa916e0a8364e5227ee661af76fde";
    hash = "sha256-1ycrr9HMrGA3ZDM8qmKcZICBupE5UShnIIhPRWdvAzA=";
  }) { inherit (stdenv.hostPlatform) system; };

  glibc_2_27 = nixpkgs_20191230.glibc.overrideAttrs (prevAttrs: {
    # Slight adjustments for compatibility with modern nixpkgs:
    #
    # - pname is required
    # - an additional getent output
    # - passthru libgcc

    pname = "glibc";

    outputs = prevAttrs.outputs ++ [ "getent" ];

    postInstall = prevAttrs.postInstall + ''
      install -Dm755 $bin/bin/getent -t $getent/bin

      # libgcc_s.so is normally a linker script that adds -lgcc for static
      # functions. However due to bootstrapping requirements, libgcc_s is
      # also added to the glibc library directory. Unfortunately, here it
      # is a symlink to libgcc_s.so.1. This breaks linkage with g++, since
      # the static library is not used. Newer glibc versions allow fixing
      # this easily by setting up the libgcc as a user-trusted directory.
      # We'll fix it here by replacing libgcc_s.so by a linker script.
      rm -f $out/lib/libgcc_s.so
      echo "GROUP ( libgcc_s.so.1 -lgcc )" > $out/lib/libgcc_s.so
    '';

    passthru = prevAttrs.passthru // {
      # Should be stdenv's gcc, but we don't have access to it.
      libgcc = stdenv.cc.cc.libgcc;
    };
  });

  stdenvWith =
    newGlibc: newGcc: stdenv:
    let
      # We need gcc to have a libgcc/libstdc++ that is compatible with
      # glibc. We do this in three steps to avoid an infinite recursion:
      # (1) we create an stdenv with gcc and glibc; (2) we rebuild gcc using
      # this stdenv, so that we have a libgcc/libstdc++ that is compatible
      # with glibc; (3) we create the final stdenv that contains the compatible
      # gcc + glibc.
      onlyGlibc = overrideCC stdenv (wrapCCWith {
        cc = newGcc;
        bintools = wrapBintoolsWith {
          bintools = bintools-unwrapped;
          libc = newGlibc;
        };
      });
      compilerWrapped = wrapCCWith rec {
        cc = newGcc.override { stdenv = onlyGlibc; };
        bintools = wrapBintoolsWith {
          bintools = bintools-unwrapped;
          libc = newGlibc;
        };
      };
    in
    overrideCC stdenv compilerWrapped;

in
stdenvWith glibc_2_27 (if cudaSupport then cudaPackages.backendStdenv else gcc13Stdenv).cc.cc stdenv
