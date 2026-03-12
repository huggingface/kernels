{
  buildPythonPackage,
  fetchFromGitHub,

  pythonRelaxDepsHook,
  setuptools,

  cuda-bindings,
  cuda-pathfinder,
}:

buildPythonPackage rec {
  pname = "cuda-python";
  version = "13.1.1";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "cuda-python";
    rev = "v${version}";
    hash = "sha256-AvIw2G4EGWsX7PxFGxzECXhwkbwYslHzRdUArTNf7jE=";
  };

  sourceRoot = "source/cuda_python";

  nativeBuildInputs = [
    pythonRelaxDepsHook
  ];

  build-system = [ setuptools ];

  dependencies = [
    cuda-bindings
    cuda-pathfinder
  ];

  pythonRelaxDeps = [ "cuda-bindings" ];
}
