{
  buildPythonPackage,
  fetchurl,

  pythonRelaxDepsHook,
  setuptools,

  cuda-python,
  numpy,
  nvidia-cuda-nvdisasm,
  protobuf6,
  typing-extensions,
}:

buildPythonPackage rec {
  pname = "nvidia-cutlass-dsl-libs-core";
  version = "4.6.1";
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/py3/n/nvidia_cutlass_dsl_libs_core/nvidia_cutlass_dsl_libs_core-${version}-py3-none-any.whl";
    hash = "sha256-8diV7iSxunEbK51KQ8Yvofb+OlBjTiXu/+iPoX9sXkc=";
  };

  dependencies = [
    cuda-python
    numpy
    nvidia-cuda-nvdisasm
    protobuf6
    typing-extensions
  ];
}
