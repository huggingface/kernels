# Vendored from nixpkgs
{
  lib,
  stdenv,
  cmake,
  jq,
  python3,
  ninja,
  pkg-config,
  rocmPackages,
  writableTmpDirAsHomeHook,
  writeShellScriptBin,
  xz,
}:

{
  version,
  gpuTargets,
  patches ? [ ],
  src,
  images,
  extraPythonDepends ? ps: [ ],
}:

let
  gpuTargets' = lib.concatStringsSep ";" gpuTargets;
  compiler = "amdclang++";
in
stdenv.mkDerivation (finalAttrs: {
  pname = "aotriton";

  inherit version src patches;

  env = {
    #CXX = compiler;
    ROCM_PATH = "${rocmPackages.clr}";
    CFLAGS = "-w -g1 -gz -Wno-c++11-narrowing";
    CXXFLAGS = finalAttrs.env.CFLAGS;

    # aotriton passes a lot of files to the linker.
    NIX_LD_USE_RESPONSE_FILE = 1;
  };

  requiredSystemFeatures = [ "big-parallel" ];

  nativeBuildInputs = [
    cmake
    jq
    rocmPackages.rocm-cmake
    pkg-config
    python3
    ninja
    rocmPackages.clr
    writableTmpDirAsHomeHook # venv wants to cache in ~
    (writeShellScriptBin "amdclang++" ''
      exec ${rocmPackages.llvm.clang}/bin/clang++ "$@"
    '')
  ];

  buildInputs = [
    rocmPackages.clr
    xz
  ]
  ++ (with python3.pkgs; [
    wheel
    packaging
    pyyaml
    numpy
    filelock
    iniconfig
    pluggy
    pybind11
    pandas
    triton
  ]);

  preConfigure = lib.optionalString (lib.versionAtLeast version "0.11.1") ''
    # Since we use pre-built images, we can grab the image SHA from there.
    # As of 0.11.1b this doesn't seem to be used for image loading yet, but
    # just in case this happens in the future, we set this to the actual
    # value and not a stub.
    export AOTRITON_CI_SUPPLIED_SHA1=$(jq -r '.["AOTRITON_GIT_SHA1"]' ${images}/lib/aotriton.images/amd-gfx90a/__signature__)

    # Need to set absolute paths to VENV and its PYTHON or
    # build fails with "AOTRITON_INHERIT_SYSTEM_SITE_TRITON is enabled
    # but triton is not available … no such file or directory"
    # Set via a preConfigure hook so a valid absolute path can be
    # picked if nix-shell is used against this package
    cmakeFlagsArray+=(                                                                                                                                                                                                               
      "-DVENV_DIR=$(pwd)/aotriton-venv/"                                                                                                                                                                                             
      "-DVENV_BIN_PYTHON=$(pwd)/aotriton-venv/bin/python"                                                                                                                                                                            
    )                                                                                                                                                                                                                                
  '';

  # From README:
  # Note: do not run ninja separately, due to the limit of the current build system,
  # ninja install will run the whole build process unconditionally.
  dontBuild = true;

  installPhase = ''
    runHook preInstall
    ninja -v install
    ln -sf ${images}/lib/aotriton.images $out/lib/aotriton.images
    runHook postInstall
  '';

  doCheck = false;
  doInstallCheck = false;

  cmakeFlags = [
    # Disable building kernels if no supported targets are enabled
    (lib.cmakeBool "AOTRITON_NOIMAGE_MODE" true)
    # Use preinstalled triton from our python's site-packages
    (lib.cmakeBool "AOTRITON_INHERIT_SYSTEM_SITE_TRITON" true)
    # Avoid kernels being skipped if build host is overloaded
    (lib.cmakeFeature "AOTRITON_GPU_BUILD_TIMEOUT" "0")
    (lib.cmakeFeature "CMAKE_CXX_COMPILER" compiler)
    # Manually define CMAKE_INSTALL_<DIR>
    # See: https://github.com/NixOS/nixpkgs/pull/197838
    (lib.cmakeFeature "CMAKE_INSTALL_BINDIR" "bin")
    (lib.cmakeFeature "CMAKE_INSTALL_LIBDIR" "lib")
    (lib.cmakeFeature "CMAKE_INSTALL_INCLUDEDIR" "include")
    (lib.cmakeFeature "AOTRITON_TARGET_ARCH" gpuTargets')
    (lib.cmakeBool "AOTRITON_USE_TORCH" false)
  ];

  meta = with lib; {
    description = "Ahead of Time (AOT) Triton Math Library";
    homepage = "https://github.com/ROCm/aotriton";
    license = with licenses; [ mit ];
    platforms = platforms.linux;
  };
})
