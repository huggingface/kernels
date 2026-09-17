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
  nvidia-cuda-nvdisasm,
  nvidia-cutlass-dsl-libs-core,
  nvidia-cutlass-dsl-libs-cu,
  protobuf6,
  typing-extensions,
}:

let
  format = "wheel";
  pyShortVersion = "cp" + builtins.replaceStrings [ "." ] [ "" ] python.pythonVersion;
  hashes = {
    cp314-x86_64-linux-cu12 = "sha256-J0Yf9SM4KAtsAkdfh4USc7WhHDh9nao4M3YiWj92fMI=";
    cp314-aarch64-linux-cu12 = "sha256-loFa+YsZuARYAv2LClPV4Zfg07zkLHA44oYa98hIP10=";
    cp314-x86_64-linux-cu13 = "sha256-J0Yf9SM4KAtsAkdfh4USc7WhHDh9nao4M3YiWj92fMI=";
    cp314-aarch64-linux-cu13 = "sha256-loFa+YsZuARYAv2LClPV4Zfg07zkLHA44oYa98hIP10=";
  };
  hash =
    hashes."${pyShortVersion}-${stdenv.system}-cu${cudaPackages.cudaMajorVersion}"
      or (throw "Unsupported Python version: ${pyShortVersion}-${stdenv.system}-cu${cudaPackages.cudaMajorVersion}");

in
buildPythonPackage rec {
  pname = "nvidia-cutlass-dsl-libs";
  version = "4.6.1";
  inherit format;

  src = fetchPypi {
    pname = "nvidia_cutlass_dsl_libs_base";
    python = pyShortVersion;
    abi = pyShortVersion;
    dist = pyShortVersion;
    platform = "manylinux_2_28_${stdenv.hostPlatform.uname.processor}";
    inherit format hash version;
  };

  nativeBuildInputs = [
    autoAddDriverRunpath
    autoPatchelfHook
    pythonRelaxWheelDepsHook
    pythonWheelDepsCheckHook
  ];

  dependencies = [
    cuda-python
    numpy
    nvidia-cuda-nvdisasm
    nvidia-cutlass-dsl-libs-core
    nvidia-cutlass-dsl-libs-cu
    protobuf6
    typing-extensions
  ];

  autoPatchelfIgnoreMissingDeps = [
    "libcuda.so.1"
  ];

  pythonRemoveDeps = [
    # nvidia-cutlass-dsl-libs-cu13 has a dependency on the base package,
    # but it has the same contents + CUDA 13 extensions.
    "nvidia-cutlass-dsl-libs-base"
  ];

  meta = {
    description = "NVIDIA CUTLASS Python DSL native libraries";
    homepage = "https://github.com/NVIDIA/cutlass";
    license = lib.licenses.unfree;
    broken = !(stdenv.hostPlatform.isLinux && cudaPackages.cudaAtLeast "12.8");
    sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
  };
}
