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
      toolchain = pkgs.stdenvNoCC.mkDerivation {
        pname = "toolchain";
        version = final.gcc-toolset-14-gcc.version;

        nativeBuildInputs = with pkgs; [ rsync ];

        dontUnpack = true;

        installPhase = with final; ''
          runHook preInstall

          mkdir $out
          for path in ${gcc-toolset-14-binutils} ${gcc-toolset-14-gcc} ${gcc-toolset-14-gcc-cxx} ${gcc-toolset-14-libstdcxx-devel} ${glibc} ${glibc-headers} ${kernel-headers}; do
            rsync --exclude=nix-support -a $path/ $out/
          done

          runHook postInstall
        '';
      };

      stdenv = pkgs.overrideCC pkgs.stdenv (
        pkgs.wrapCCWith {
          cc = final.toolchain;
          libc = final.glibc;
          bintools = pkgs.wrapBintoolsWith {
            bintools = final.toolchain;
            libc = final.glibc;
          };
        }
      );
    })
  ];
in
lib.makeScope newScope (lib.extends composed fixedPoint)
