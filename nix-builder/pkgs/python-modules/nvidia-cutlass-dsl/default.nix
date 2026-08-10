{
  lib,
  fetchPypi,
  python,

  buildPythonPackage,
  pythonWheelDepsCheckHook,

  cudaPackages,

  nvidia-cutlass-dsl-libs,
}:

let
  format = "wheel";
in
buildPythonPackage rec {
  pname = "nvidia-cutlass-dsl";
  version = "4.6.1";
  inherit format;

  src = fetchPypi {
    inherit format version;
    pname = "nvidia_cutlass_dsl";
    dist = "py3";
    python = "py3";
    hash = "sha256-kxNanUjhvt9YSCjgoCHxdKwxWR7yH26lPCBhacy/qyY=";
  };

  nativeBuildInputs = [
    pythonWheelDepsCheckHook
  ];

  dependencies = [
    nvidia-cutlass-dsl-libs
  ];

  pythonRemoveDeps = lib.optionals (cudaPackages.cudaAtLeast "13.0") [
    "nvidia-cutlass-dsl-libs-cu12"
  ];

  pythonImportsCheck = [ "nvidia_cutlass_dsl" ];

  meta = {
    broken = nvidia-cutlass-dsl-libs.meta.broken;
  };
}
