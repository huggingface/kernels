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
          x86_64-linux = "sha256-2jdFj4xQdNWQQN7dvQBUYMDPcNHvkN19LIFuRoYDjoc=";
          aarch64-linux = "sha256-4G0nKyUUnhqDtEb/Pb789vgvZoZcSejZ23VoO7z/C18=";
        };
      };
      cuda_13 = {
        version = "13.2.0";
        hash = {
          x86_64-linux = "sha256-fcoNoFPTtMxIae/0nGHAPzxduqC81xIxejWNW48/OF0=";
          aarch64-linux = "sha256-ZinKLfb3lbeEdSQJvK7b0ip6ZRt0tWoWXrwMncvVBNA=";
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
