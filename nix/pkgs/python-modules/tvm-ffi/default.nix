{
  buildPythonPackage,
  fetchFromGitHub,

  cmake,
  cython,
  ninja,
  scikit-build-core,
  setuptools-scm,
}:

buildPythonPackage rec {
  pname = "tvm-ffi";
  version = "0.1.8-post2";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "apache";
    repo = "tvm-ffi";
    rev = "v${version}";
    hash = "sha256-IPt1/VokZ4STfnt+JEYaCkslOWFCHZYWqpFCbJtvGQY=";
    fetchSubmodules = true;
  };

  build-system = [
    cmake
    cython
    ninja
    scikit-build-core
    setuptools-scm
  ];

  dontUseCmakeConfigure = true;
}
