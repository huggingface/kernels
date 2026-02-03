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
        version = "12.9.4";
        hash = "sha256-Mr3Fp2kGvkxh65j1RqZ4bFdzqIHzsWZIZEm10UHko58=";
      };
    in
    {
      "12.6" = cuda_12;
      "12.8" = cuda_12;
      "12.9" = cuda_12;
      "13.0" = {
        version = "13.0.3";
        hash = "sha256-US0NgDpeR6ikLVo0zgkygCv3L+lS/bEax5hxWjXG5cs=";
      };
    };

  versionHash =
    versionHashes.${cudaPackages.cudaMajorMinorVersion}
      or (throw "Unsupported CUDA version: ${cudaPackages.cudaMajorMinorVersion}");
  inherit (versionHash) hash version;

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
}
