{
  lib,
  fetchPypi,
  buildPythonPackage,

  absl-py,
  numpy,
}:

buildPythonPackage rec {
  pname = "einshape";
  version = "1.0";
  format = "wheel";

  src = fetchPypi {
    inherit format pname version;
    dist = "py3";
    python = "py3";
    hash = "sha256-QtpMLeo6J/h+5Fp87lBypja5fLGEuwe/XWQSug/3uWU=";
  };

  dependencies = [
    absl-py
    numpy
  ];

  pythonImportsCheck = [ "einshape" ];

  meta = with lib; {
    description = "DSL-based reshaping library for JAX and other frameworks";
    homepage = "https://github.com/google-deepmind/einshape";
    license = licenses.asl20;
  };
}
