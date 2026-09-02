{
  lib,
  stdenv,
  makeWrapper,
  markForRocmRootHook,
  rsync,
  clang,
  amdrocm-runtime,
  amdrocm-opencl,
  setupRocmHook,
}:

let
  hipClangPath = "${clang}/bin";
  wrapperArgs = [
    "--prefix PATH : $out/bin"
    "--prefix LD_LIBRARY_PATH : ${amdrocm-runtime}"
    "--set HIP_PLATFORM amd"
    "--set HIP_PATH $out"
    "--set HIP_CLANG_PATH ${hipClangPath}"
    "--set DEVICE_LIB_PATH ${amdrocm-runtime}/amdgcn/bitcode"
    "--set HSA_PATH ${amdrocm-runtime}"
    "--set ROCM_PATH $out"
  ];
in
stdenv.mkDerivation {
  pname = "rocm-clr";
  version = amdrocm-runtime.version;

  nativeBuildInputs = [
    markForRocmRootHook
    makeWrapper
    rsync
  ];

  propagatedBuildInputs = [
    setupRocmHook
  ];

  dontUnpack = true;

  installPhase = ''
    runHook preInstall

    mkdir -p $out

    for path in ${amdrocm-runtime} ${amdrocm-opencl}; do
      rsync -a --exclude=nix-support $path/ $out/
    done

    chmod -R u+w $out

    wrapProgram $out/bin/hipcc ${lib.concatStringsSep " " wrapperArgs}
    wrapProgram $out/bin/hipconfig ${lib.concatStringsSep " " wrapperArgs}

    runHook postInstall
  '';

  postInstall = ''
    mkdir -p $out/nix-support/
    echo '
    export HIP_PATH="${placeholder "out"}"
    export HIP_PLATFORM=amd
    export HIP_DEVICE_LIB_PATH="${amdrocm-runtime}/amdgcn/bitcode"
    export HIP_CLANG_PATH="${hipClangPath}"
    export HSA_PATH="${amdrocm-runtime}"' > $out/nix-support/setup-hook

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
      "950"
      "1010"
      "1012"
      "1030"
      "1100"
      "1101"
      "1102"
      "1150"
      "1151"
      "1152"
      "1153"
      "1200"
      "1201"
      "1250"
    ] (target: "gfx${target}");
  };

}
