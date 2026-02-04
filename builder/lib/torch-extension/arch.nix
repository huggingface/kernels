{
  cudaSupport ? torch.cudaSupport,
  rocmSupport ? torch.rocmSupport,
  xpuSupport ? torch.xpuSupport,

  lib,
  pkgs,
  stdenv,
  writeText,

  # Native build inputs
  build2cmake,
  cmake,
  cmakeNvccThreadsHook,
  cuda_nvcc,
  get-kernel-check,
  kernel-abi-check,
  kernel-layout-check,
  ninja,
  python3,
  remove-bytecode-hook,
  rewrite-nix-paths-macho,
  writeScriptBin,

  # Framework packages
  cudaPackages,
  rocmPackages,
  xpuPackages,

  # Build inputs
  apple-sdk_26,
  clr,
  oneapi-torch-dev,
  onednn-xpu,
  torch,
}:

{
  buildConfig,

  # Whether to do ABI checks.
  doAbiCheck ? true,

  # Whether to run get-kernel-check.
  doGetKernelCheck ? true,

  kernelName,

  # Extra dependencies (such as CUTLASS).
  extraDeps ? [ ],

  nvccThreads,

  # A stringly-typed list of Python dependencies. Ideally we'd take a
  # list of derivations, but we also need to write the dependencies to
  # the output.
  pythonDeps,

  backendPythonDeps,

  # Wheter to strip rpath for non-nix use.
  stripRPath ? false,

  # Revision to bake into the ops name.
  rev,

  src,
}:

# Extra validation - the environment should correspind to the build config.
assert (buildConfig ? cudaVersion) -> cudaSupport;
assert (buildConfig ? rocmVersion) -> rocmSupport;
assert (buildConfig ? xpuVersion) -> xpuSupport;
assert (buildConfig.metal or false) -> stdenv.hostPlatform.isDarwin;

let
  inherit (import ../deps.nix { inherit lib pkgs torch; }) resolvePythonDeps resolveBackendPythonDeps;

  dependencies =
    resolvePythonDeps pythonDeps
    ++ resolveBackendPythonDeps buildConfig.backend backendPythonDeps
    ++ [ torch ];

  moduleName = builtins.replaceStrings [ "-" ] [ "_" ] kernelName;

  # On Darwin, we need the host's xcrun for `xcrun metal` to compile Metal shaders.
  # It's not supported by the nixpkgs shim.
  xcrunHost = writeScriptBin "xcrunHost" ''
    # Use system SDK for Metal files.
    unset DEVELOPER_DIR
    /usr/bin/xcrun $@
  '';

  metalSupport = buildConfig.metal or false;

in

stdenv.mkDerivation (prevAttrs: {
  name = "${kernelName}-torch-ext";

  inherit
    doAbiCheck
    moduleName
    nvccThreads
    src
    ;

  # Generate build files.
  postPatch = ''
    build2cmake generate-torch \
      --ops-id ${rev} build.toml
  '';

  preConfigure =
    # This is a workaround for https://openradar.appspot.com/FB20389216 - even
    # if the user downloaded the Metal toolchain, the mapping is not set up
    # for the Nix build users. To make things worse, we cannot set up a mapping
    # because the Nix build users do not have a writable home directory and
    # showComponent/downloadComponent do not respect the HOME variable. So
    # instead, we'll use showComponent (which will emit a lot of warnings due
    # to the above) to grab the path of the Metal toolchain.
    lib.optionalString metalSupport ''
      METAL_PATH=$(${xcrunHost}/bin/xcrunHost xcodebuild -showComponent MetalToolchain 2> /dev/null | sed -rn "s/Toolchain Search Path: (.*)/\1/p")
      if [ ! -d "$METAL_PATH" ]; then
        >&2 echo "Cannot find Metal toolchain, use: xcodebuild -downloadComponent MetalToolchain"
        exit 1
      fi

      cmakeFlagsArray+=("-DMETAL_TOOLCHAIN=$METAL_PATH/Metal.xctoolchain")
    '';

  # hipify copies files, but its target is run in the CMake build and install
  # phases. Since some of the files come from the Nix store, this fails the
  # second time around.
  preInstall = ''
    chmod -R u+w .
  '';

  nativeBuildInputs = [
    cmake
    ninja
    build2cmake
    kernel-abi-check
    kernel-layout-check
    remove-bytecode-hook
  ]
  ++ lib.optionals doGetKernelCheck [
    (get-kernel-check.override { python3 = python3.withPackages (ps: dependencies); })
  ]
  ++ lib.optionals cudaSupport [
    cmakeNvccThreadsHook
    cuda_nvcc
  ]
  ++ lib.optionals rocmSupport [
    clr
  ]
  ++ lib.optionals xpuSupport ([
    xpuPackages.ocloc
    oneapi-torch-dev
  ])
  ++ lib.optionals stdenv.hostPlatform.isDarwin [
    rewrite-nix-paths-macho
  ];

  buildInputs = [
    torch
    torch.cxxdev
  ]
  ++ lib.optionals cudaSupport (
    with cudaPackages;
    [
      cuda_cudart

      # Make dependent on build configuration dependencies once
      # the Torch dependency is gone.
      cuda_cccl
      libcublas
      libcusolver
      libcusparse
    ]
  )
  ++ lib.optionals rocmSupport (
    with rocmPackages;
    [
      hipcub-devel
      hipsparselt
      rocprim-devel
      rocthrust-devel
      rocwmma-devel
    ]
  )
  ++ lib.optionals xpuSupport ([
    oneapi-torch-dev
    onednn-xpu
  ])
  ++ lib.optionals stdenv.hostPlatform.isDarwin [
    apple-sdk_26
  ]
  ++ extraDeps;

  env =
    lib.optionalAttrs cudaSupport {
      CUDAToolkit_ROOT = "${lib.getDev cudaPackages.cuda_nvcc}";
      TORCH_CUDA_ARCH_LIST = lib.concatStringsSep ";" torch.cudaCapabilities;
    }
    // lib.optionalAttrs rocmSupport {
      PYTORCH_ROCM_ARCH = lib.concatStringsSep ";" torch.rocmArchs;
    }
    // lib.optionalAttrs xpuSupport {
      MKLROOT = oneapi-torch-dev;
      SYCL_ROOT = oneapi-torch-dev;
    };

  # If we use the default setup, CMAKE_CUDA_HOST_COMPILER gets set to nixpkgs g++.
  dontSetupCUDAToolkitCompilers = true;

  cmakeFlags = [
    (lib.cmakeBool "BUILD_ALL_SUPPORTED_ARCHS" true)
    (lib.cmakeFeature "Python_EXECUTABLE" "${python3.withPackages (ps: [ torch ])}/bin/python")
    # Fix: file RPATH_CHANGE could not write new RPATH, we are rewriting
    # rpaths anyway.
    (lib.cmakeBool "CMAKE_SKIP_RPATH" true)
  ]
  ++ lib.optionals cudaSupport [
    (lib.cmakeFeature "CMAKE_CUDA_HOST_COMPILER" "${stdenv.cc}/bin/g++")
  ]
  ++ lib.optionals rocmSupport [
    # Ensure sure that we use HIP from our CLR override and not HIP from
    # the symlink-joined ROCm toolkit.
    (lib.cmakeFeature "CMAKE_HIP_COMPILER_ROCM_ROOT" "${clr}")
    (lib.cmakeFeature "HIP_ROOT_DIR" "${clr}")
  ]
  ++ lib.optionals metalSupport [
    # Use host compiler for Metal. Not included in the redistributable SDK.
    # Re-enable when the issue mentioned in preConfigure is solved.
    #(lib.cmakeFeature "METAL_COMPILER" "${xcrunHost}/bin/xcrunHost")
  ];

  postInstall =
    let
      buildVariant = torch.variant;
    in
    ''
      rm -rf $out/_${moduleName}_${rev}
    ''
    + (lib.optionalString (stripRPath && stdenv.hostPlatform.isLinux)) ''
      find $out/ -name '*.so' \
        -exec patchelf --set-rpath "" {} \;
    ''
    + (lib.optionalString (stripRPath && stdenv.hostPlatform.isDarwin)) ''
      find $out/ -name '*.so' \
        -exec rewrite-nix-paths-macho {} \;

      # Stub some rpath.
      find $out/ -name '*.so' \
        -exec install_name_tool -add_rpath "@loader_path/lib" {} \;
    '';

  doInstallCheck = true;

  # We need access to the host system on Darwin for the Metal compiler.
  __noChroot = metalSupport;

  passthru = {
    inherit dependencies torch;
    inherit (torch) variant;
  };
})
