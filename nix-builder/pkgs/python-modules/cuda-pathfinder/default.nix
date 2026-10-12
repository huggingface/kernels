{
  buildPythonPackage,
  fetchFromGitHub,
  setuptools,
  setuptools-scm,
}:

buildPythonPackage rec {
  pname = "cuda-pathfinder";
  version = "1.4.1";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "cuda-python";
    tag = "${pname}-v${version}";
    hash = "sha256-rdDJGkA+20NB5dvwJHOSUGBzvx3lfLuuUqMhvz1dmXY=";
  };

  sourceRoot = "source/cuda_pathfinder";

  build-system = [
    setuptools
    setuptools-scm
  ];

  # The source tarball has no git metadata for setuptools-scm.
  env.SETUPTOOLS_SCM_PRETEND_VERSION = version;

  pythonImportsCheck = [ "cuda.pathfinder" ];
}
