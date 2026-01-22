{
  stdenv,
  fetchPypi,
  python,

  buildPythonPackage,
  autoPatchelfHook,
  autoAddDriverRunpath,
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
    cp313-x86_64-linux = "sha256-Lm7cFGjZjhdg4YXftwjcmaAVThSg9IOejUkc61YzoZk=";
  };
  hash =
    hashes."${pyShortVersion}-${stdenv.system}"
      or (throw "Unsupported Python version: ${pyShortVersion}-${stdenv.system}");

in
buildPythonPackage rec {
  pname = "nvidia-cutlass-dsl";
  version = "4.3.0";
  inherit format;

  src = fetchPypi {
    pname = "nvidia_cutlass_dsl";
    python = pyShortVersion;
    abi = pyShortVersion;
    dist = pyShortVersion;
    platform = "manylinux_2_28_${stdenv.hostPlatform.uname.processor}";
    inherit format hash version;
  };

  nativeBuildInputs = [
    autoAddDriverRunpath
    autoPatchelfHook
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

  meta = {
    broken = !(cudaPackages.cudaAtLeast "12.8");
  };
}
