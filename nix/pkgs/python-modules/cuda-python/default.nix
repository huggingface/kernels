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
  version = "13.1.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "cuda-python";
    rev = "v${version}";
    hash = "sha256-aF06WwThJmGEqUcVP4dooym1uqYjUM45arqZaxjlTuA=";
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
