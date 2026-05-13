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

    (final: prev: {
      gcc-unwrapped = pkgs.stdenvNoCC.mkDerivation {
        pname = "gcc";
        version = final.gcc-toolset-14-gcc.version;

        nativeBuildInputs = with pkgs; [ rsync ];

        dontUnpack = true;

        installPhase = with final; ''
          runHook preInstall

          mkdir $out
          for path in ${gcc-toolset-14-gcc} ${gcc-toolset-14-gcc-cxx} ${gcc-toolset-14-libstdcxx-devel} ${glibc-headers} ${kernel-headers} ${libgcc} ${libstdcxx}; do
            rsync --exclude=nix-support -a $path/ $out/
          done

          chmod -R u+w $out

          # Move around libraries to reflect what Nix expects for gccForLibs.
          mv $out/lib/gcc/${pkgs.stdenv.hostPlatform.linuxArch}-redhat-linux/14/{libstdc++*,libgcc_s*,libgomp*} $out/lib

          # Update linker script with Nix paths.
          substituteInPlace $out/lib/libstdc++.so \
            --replace-fail "/usr/lib64/libstdc++.so.6" "$out/lib/libstdc++.so.6"
          substituteInPlace $out/lib/libgcc_s.so \
            --replace-fail "/lib64/libgcc_s.so.1" "$out/lib/libgcc_s.so.1"

          runHook postInstall
        '';

      };

      binutils = pkgs.wrapBintoolsWith {
        bintools = final.gcc-toolset-14-binutils;
        libc = glibc_2_28;
      };

      gcc = pkgs.wrapCCWith {
        bintools = final.binutils;
        coreutils = final.coreutils;
        cc = final.gcc-unwrapped;
        gccForLibs = final.gcc-unwrapped;
        libc = glibc_2_28;
        isGNU = true;
        useCcForLibs = true;
      };

      stdenv = pkgs.overrideCC pkgs.stdenv final.gcc;
    })
  ];
in
lib.makeScope newScope (lib.extends composed fixedPoint)
