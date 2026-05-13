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
          mv $out/lib/gcc/${stdenv.hostPlatform.linuxArch}-redhat-linux/14/{libstdc++*,libgcc_s*,libgomp*} $out/lib

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
        libc = final.glibc;
      };

      gcc = pkgs.wrapCCWith {
        bintools = final.binutils;
        coreutils = final.coreutils;
        cc = final.gcc-unwrapped;
        gccForLibs = final.gcc-unwrapped;
        libc = final.glibc;
        isGNU = true;
        useCcForLibs = true;
      };

      stdenv = pkgs.overrideCC pkgs.stdenv final.gcc;
    })
  ];
in
lib.makeScope newScope (lib.extends composed fixedPoint)
