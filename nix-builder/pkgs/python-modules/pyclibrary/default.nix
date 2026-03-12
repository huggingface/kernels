{
  buildPythonPackage,
  fetchFromGitHub,

  setuptools,
  setuptools-scm,

  pyparsing,
}:

buildPythonPackage rec {
  pname = "pyclibrary";
  version = "0.3.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "MatthieuDartiailh";
    repo = "pyclibrary";
    tag = version;
    hash = "sha256-RyIbRySRWSZwKP5G6yXYCOnfKOV0165aPyjMf3nSbOM=";
  };

  build-system = [
    setuptools
    setuptools-scm
  ];

  dependencies = [ pyparsing ];

  pythonImportsCheck = [ "pyclibrary" ];
}
