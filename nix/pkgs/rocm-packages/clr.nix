{
  lib,
  stdenv,
  makeWrapper,
  markForRocmRootHook,
  rsync,
  clang,
  comgr,
  hipcc,
  hip-devel,
  hip-runtime-amd,
  hsa-rocr,
  perl,
  rocm,
  rocm-core,
  rocm-device-libs,
  rocm-opencl,
  rocminfo,
  setupRocmHook,
}:

let
  hipClangPath = "${clang}/bin";
  wrapperArgs = [
    "--prefix PATH : $out/bin"
    "--prefix LD_LIBRARY_PATH : ${hsa-rocr}"
    "--set HIP_PLATFORM amd"
    "--set HIP_PATH $out"
    "--set HIP_CLANG_PATH ${hipClangPath}"
    "--set DEVICE_LIB_PATH ${rocm-device-libs}/amdgcn/bitcode"
    "--set HSA_PATH ${hsa-rocr}"
    "--set ROCM_PATH $out"
  ];
in
stdenv.mkDerivation {
  pname = "rocm-clr";
  version = rocm.version;

  nativeBuildInputs = [
    markForRocmRootHook
    makeWrapper
    rsync
  ];

  propagatedBuildInputs = [
    comgr
    rocm-device-libs
    hsa-rocr
    perl
    rocminfo
    setupRocmHook
  ];

  dontUnpack = true;

  installPhase = ''
    runHook preInstall

    mkdir -p $out

    for path in ${hipcc} ${hip-devel} ${hip-runtime-amd} ${rocm-core} ${rocm-opencl}; do
      rsync -a --exclude=nix-support $path/ $out/
    done

    chmod -R u+w $out

    # Some build infra expects rocminfo to be in the clr package. Easier
    # to just symlink it than to patch everything.
    ln -s ${rocminfo}/bin/* $out/bin

    wrapProgram $out/bin/hipcc ${lib.concatStringsSep " " wrapperArgs}
    wrapProgram $out/bin/hipconfig ${lib.concatStringsSep " " wrapperArgs}

    # Removed in ROCm 7.
    if [ -f $out/bin/hipcc.pl ]; then
      wrapProgram $out/bin/hipcc.pl ${lib.concatStringsSep " " wrapperArgs}
    fi
    if [ -f $out/bin/hipconfig.pl ]; then
      wrapProgram $out/bin/hipconfig.pl ${lib.concatStringsSep " " wrapperArgs}
    fi

    runHook postInstall
  '';

  postInstall = ''
    mkdir -p $out/nix-support/
    echo '
    export HIP_PATH="${placeholder "out"}"
    export HIP_PLATFORM=amd
    export HIP_DEVICE_LIB_PATH="${rocm-device-libs}/amdgcn/bitcode"
    export HIP_CLANG_PATH="${hipClangPath}"
    export HSA_PATH="${hsa-rocr}"' > $out/nix-support/setup-hook

    ln -s ${clang} $out/llvm
  '';

  dontStrip = true;

  passthru = {
    gpuTargets = lib.forEach [
      "803"
      "900"
      "906"
      "908"
      "90a"
      "940"
      "941"
      "942"
      "1010"
      "1012"
      "1030"
      "1100"
      "1101"
      "1102"
    ] (target: "gfx${target}");
  };

}
