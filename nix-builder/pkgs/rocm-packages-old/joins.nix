{
  lib,
  stdenv,
  makeWrapper,
  rsync,
}:

final: prev: {
  clr = final.callPackage ./clr.nix {
    inherit (final)
      comgr
      hipcc
      hip-devel
      hip-runtime-amd
      hsa-rocr
      markForRocmRootHook
      rocm-device-libs
      rocm-opencl
      rocminfo
      setupRocmHook
      ;
    inherit (final.llvm) clang;
  };

  openmp = stdenv.mkDerivation {
    pname = "rocm-openmp";
    version = final.hipcc.version;

    nativeBuildInputs = [
      final.markForRocmRootHook
      makeWrapper
      rsync
    ];

    dontUnpack = true;

    installPhase = with final; ''
      runHook preInstall

      mkdir -p $out

      for path in ${openmp-extras-devel} ${openmp-extras-runtime}; do
        rsync --exclude=nix-support -a $path/lib/llvm/ $out/
      done

      runHook postInstall
    '';

    dontStrip = true;
  };
}
