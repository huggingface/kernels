{
  buildPythonPackage,
  fetchFromGitHub,
  setuptools,
}:

buildPythonPackage rec {
  pname = "cuda-pathfinder";
  version = "1.3.3";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "cuda-python";
    rev = "${pname}-v${version}";
    hash = "sha256-aF06WwThJmGEqUcVP4dooym1uqYjUM45arqZaxjlTuA=";
  };

  sourceRoot = "source/cuda_pathfinder";

  build-system = [ setuptools ];

  pythonImportsCheck = [ "cuda.pathfinder" ];
}
