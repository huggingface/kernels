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
  typing-extensions,
}:

let
  format = "wheel";
  pyShortVersion = "cp" + builtins.replaceStrings [ "." ] [ "" ] python.pythonVersion;
  hashes = {
    cp313-x86_64-linux-cu12 = "sha256-CaaqzoQVlOo5z3X3zuaNwxuM88263/y9SU8I/HPGRm0=";
    cp313-aarch64-linux-cu12 = "sha256-fKTlypvQyqLseX67HkBLs+sPdtI1gjYReVyxTY00Hkw=";
    cp313-x86_64-linux-cu13 = "sha256-famVhPTw1JjKcKZHAiqylcpUoglJRBGBzQwyiVhNqKM=";
    cp313-aarch64-linux-cu13 = "sha256-gTcFEd/Z4cCNojAs8UD0reyJ31BOw0iiOwiqAXRyDsM=";
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
    pname =
      if cudaPackages.cudaAtLeast "13.0" then
        "nvidia_cutlass_dsl_libs_cu13"
      else
        "nvidia_cutlass_dsl_libs_base";
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
