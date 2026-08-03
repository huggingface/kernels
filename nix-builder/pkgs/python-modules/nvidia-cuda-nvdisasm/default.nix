{
  lib,
  stdenv,
  fetchPypi,
  python,

  buildPythonPackage,
  autoPatchelfHook,
  autoAddDriverRunpath,
  pythonRelaxWheelDepsHook,
  pythonWheelDepsCheckHook,

  cudaPackages,
  cuda-python,
  numpy,
  protobuf6,
  typing-extensions,
}:

let
  format = "wheel";
  pyShortVersion = "cp" + builtins.replaceStrings [ "." ] [ "" ] python.pythonVersion;
  hashes = {
    cp313-aarch64-linux = "sha256-3UdRiE+QFrm22/AHq961aB0KLtxzHdPS/anW2Hjoj3M=";
    cp313-x86_64-linux = "sha256-+hcISwfA3KaKQokvdxtLG0D76bkWYCCWI+Yc6mEcrow=";
  };
  hash =
    hashes."${pyShortVersion}-${stdenv.system}"
      or (throw "Unsupported Python version: ${pyShortVersion}-${stdenv.system}");
  processor = stdenv.hostPlatform.uname.processor;

in
buildPythonPackage rec {
  pname = "nvidia-cuda-nvdisasm";
  version = "13.3.73";
  inherit format;

  src = fetchPypi {
    pname = "nvidia_cuda_nvdisasm";
    dist = "py3";
    python = "py3";
    abi = "none";
    platform = "manylinux2014_${processor}.manylinux_2_17_${processor}";
    inherit format hash version;
  };

  nativeBuildInputs = [
    autoAddDriverRunpath
    autoPatchelfHook
    pythonWheelDepsCheckHook
  ];

  autoPatchelfIgnoreMissingDeps = [
    "libcuda.so.1"
  ];

  meta = {
    description = "Extract information from standalone cubin files";
    homepage = "https://developer.nvidia.com/cuda";
    license = lib.licenses.unfree;
    broken = !(stdenv.hostPlatform.isLinux && cudaPackages.cudaAtLeast "12.8");
    sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
  };
}
