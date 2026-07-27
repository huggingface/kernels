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
  protobuf6,
  typing-extensions,
}:

let
  cudaMajor = cudaPackages.cudaMajorVersion;
  format = "wheel";
  pyShortVersion = "cp" + builtins.replaceStrings [ "." ] [ "" ] python.pythonVersion;
  hashes = {
    cp313-x86_64-linux-cu12 = "";
    cp313-aarch64-linux-cu12 = "sha256-F9nVvbj1MCZtyEJCNqgFI+4qmAfoUqJMIglkOKEt870=";
    cp313-x86_64-linux-cu13 = "";
    cp313-aarch64-linux-cu13 = "sha256-gTcFEd/Z4cCNojAs8UD0reyJ31BOw0iiOwiqAXRyDsM=";
  };
  hash =
    hashes."${pyShortVersion}-${stdenv.system}-cu${cudaMajor}"
      or (throw "Unsupported Python version: ${pyShortVersion}-${stdenv.system}-cu${cudaMajor}");

in
buildPythonPackage rec {
  pname = "nvidia-cutlass-dsl-libs-cu${cudaMajor}";
  version = "4.6.1";
  inherit format;

  src = fetchPypi {
    pname = "nvidia_cutlass_dsl_libs_cu${cudaMajor}";
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
