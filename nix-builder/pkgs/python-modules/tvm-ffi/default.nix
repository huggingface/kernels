{
  callPackage,
  buildPythonPackage,
  fetchFromGitHub,

  cmake,
  cython,
  ninja,
  python,
  scikit-build-core,
  setuptools-scm,
}:

buildPythonPackage rec {
  pname = "apache-tvm-ffi";
  version = "0.1.9";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "apache";
    repo = "tvm-ffi";
    rev = "v${version}";
    hash = "sha256-XnlM//WW2TbjbmzYBq6itJQ7R3J646UMVQUVhV5Afwc=";
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

  postInstall = ''
    ln -s $out/${python.sitePackages}/tvm_ffi/share $out/share
  '';
}
