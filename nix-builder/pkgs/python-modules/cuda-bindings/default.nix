{
  stdenv,
  lib,
  buildPythonPackage,
  fetchPypi,
  python,
  symlinkJoin,

  autoAddDriverRunpath,
  autoPatchelfHook,
  pythonWheelDepsCheckHook,

  cuda-pathfinder,
  cudaPackages,
}:
let
  versionHashes =
    let
      cuda_12 = {
        version = "12.9.5";
        hash = {
          x86_64-linux = "sha256-QNx5yMv2Y+JNJgeyqXl4fg1oQLYkpNpIzVMDrLhIKtI=";
          aarch64-linux = "sha256-OzqJ/M1swexTWpQLQyQ3YVN65XPZd8PNGOOPhB4fioY=";
        };
      };
      cuda_13 = {
        version = "13.2.0";
        hash = {
          x86_64-linux = "sha256-9K+fPhvmA/oS1a1s/KeETJ0jC++peStavffdeZecNiY=";
          aarch64-linux = "sha256-pkZLMPRmktbH9l1KDgRQ2B3SneOvwbtRVlOXPQHCzW4=";
        };
      };
    in
    {
      "12.6" = cuda_12;
      "12.8" = cuda_12;
      "12.9" = cuda_12;
      "13.0" = cuda_13;
      "13.1" = cuda_13;
      "13.2" = cuda_13;
      "13.4" = cuda_13;
    };

  versionHash =
    versionHashes.${cudaPackages.cudaMajorMinorVersion}
      or (throw "Unsupported CUDA version: ${cudaPackages.cudaMajorMinorVersion}");
  inherit (versionHash) version;
  hash =
    versionHash.hash.${stdenv.hostPlatform.system}
      or (throw "No hash defined for system: ${stdenv.hostPlatform.system}");

  format = "wheel";
  pyShortVersion = "cp" + builtins.replaceStrings [ "." ] [ "" ] python.pythonVersion;

  cudaMajor = lib.versions.major cudaPackages.cudaMajorMinorVersion;

  # The extension modules load CUDA libraries at runtime through
  # cuda-pathfinder, add dependencies so that they can be loaded
  # and lacated through rpath.
  linkedLibraries = {
    "_bindings/cynvrtc" = "libnvrtc.so.${cudaMajor}";
    "_internal/cufile" = "libcufile.so.0";
    "_internal/nvjitlink" = "libnvJitLink.so.${cudaMajor}";
    "_internal/nvvm" = "libnvvm.so.4";
  }
  // lib.optionalAttrs (cudaMajor == "12") {
    "_internal/_nvml" = "libnvidia-ml.so.1";
  }
  // lib.optionalAttrs (cudaMajor != "12") {
    "_internal/nvfatbin" = "libnvfatbin.so.${cudaMajor}";
    "_internal/nvml" = "libnvidia-ml.so.1";
  };
in
buildPythonPackage {
  inherit format;
  pname = "cuda-bindings";
  inherit version;

  src = fetchPypi {
    pname = "cuda_bindings";
    python = pyShortVersion;
    abi = pyShortVersion;
    dist = pyShortVersion;
    platform = "manylinux_2_24_${stdenv.hostPlatform.uname.processor}.manylinux_2_28_${stdenv.hostPlatform.uname.processor}";
    inherit format hash version;
  };

  nativeBuildInputs = [
    # autoAddDriverRunpath must come after autoPatchelfHook, because
    # autoPatchelf rewrites the runpath and would drop the driver path.
    autoPatchelfHook
    autoAddDriverRunpath
    pythonWheelDepsCheckHook
  ];

  buildInputs = map lib.getLib (
    with cudaPackages;
    [
      cuda_nvrtc
      libcufile
      libnvjitlink
    ]
    ++ lib.optionals (cudaMajor != "12") [ libnvfatbin ]
  );

  # libnvidia-ml is part of the driver, found through the driver runpath.
  autoPatchelfIgnoreMissingDeps = [ "libnvidia-ml.so.1" ];

  postInstall = ''
    ${lib.concatStrings (
      lib.mapAttrsToList (module: soname: ''
        patchelf --add-needed ${soname} \
          $out/${python.sitePackages}/cuda/bindings/${module}.*.so
      '') linkedLibraries
    )}
  '';

  # libnvvm is in a subdirectory of cuda_nvcc.
  preFixup = ''
    addAutoPatchelfSearchPath --no-recurse ${lib.getLib cudaPackages.cuda_nvcc}/nvvm/lib
  '';

  dependencies = [ cuda-pathfinder ];

  pythonImportsCheck = [ "cuda.bindings" ];

  meta = {
    description = "Python bindings for CUDA";
    homepage = "https://github.com/NVIDIA/cuda-python";
    license = lib.licenses.unfreeRedistributable;
    sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
  };
}
