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
  version = "4.5.0";
  inherit format;

  src = fetchPypi {
    inherit format version;
    pname = "nvidia_cutlass_dsl";
    dist = "py3";
    python = "py3";
    hash = "sha256-OwUf4CymlCKrhA5k2YZWZ6uiiKOYSnykzNA4qCrvE0Q=";
  };

  nativeBuildInputs = [
    pythonWheelDepsCheckHook
  ];

  dependencies = [
    nvidia-cutlass-dsl-libs
  ];

  pythonRemoveDeps = lib.optionals (cudaPackages.cudaAtLeast "13.0") [
    # nvidia-cutlass-dsl-libs-cu13 has a dependency on the base package,
    # but it has the same contents + CUDA 13 extensions.
    "nvidia-cutlass-dsl-libs-base"
  ];

  meta = {
    broken = nvidia-cutlass-dsl-libs.meta.broken;
  };
}
