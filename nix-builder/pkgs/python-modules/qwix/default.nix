{
  lib,
  fetchPypi,
  buildPythonPackage,

  absl-py,
  flax,
  jax,
  jaxlib,
  numpy,
  opt-einsum,
}:

buildPythonPackage rec {
  pname = "qwix";
  version = "0.1.8";
  format = "wheel";

  src = fetchPypi {
    inherit format pname version;
    dist = "py3";
    python = "py3";
    hash = "sha256-D0aG9/EKEWh4f6xPuvTZiDvLj3igHy/xzTg005+FmxI=";
  };

  dependencies = [
    absl-py
    flax
    jax
    jaxlib
    numpy
    opt-einsum
  ];

  pythonImportsCheck = [ "qwix" ];

  meta = with lib; {
    description = "Quantization library for JAX";
    homepage = "https://github.com/google/qwix";
    license = licenses.asl20;
  };
}
