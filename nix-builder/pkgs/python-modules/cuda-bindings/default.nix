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
    autoAddDriverRunpath
    autoPatchelfHook
    pythonWheelDepsCheckHook
  ];

  dependencies = [ cuda-pathfinder ];

  pythonImportsCheck = [ "cuda.bindings" ];

  meta = {
    description = "Python bindings for CUDA";
    homepage = "https://github.com/NVIDIA/cuda-python";
    license = lib.licenses.unfreeRedistributable;
    sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
  };
}
