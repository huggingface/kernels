{
  lib,
  fetchPypi,
  python,

  buildPythonPackage,
  pythonWheelDepsCheckHook,

  cudaPackages,

  nvidia-cutlass-dsl-libs,

  # Versioned so several CuteDSL releases can coexist (same pattern as the C++ `cutlass_*`
  # attributes): kernels pick one via `python-depends`, nobody is migrated implicitly.
  version ? "4.5.0",
  hash ? "sha256-OwUf4CymlCKrhA5k2YZWZ6uiiKOYSnykzNA4qCrvE0Q=",
}:

let
  format = "wheel";
in
buildPythonPackage rec {
  pname = "nvidia-cutlass-dsl";
  inherit format version;

  src = fetchPypi {
    inherit format version;
    pname = "nvidia_cutlass_dsl";
    dist = "py3";
    python = "py3";
    inherit hash;
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
