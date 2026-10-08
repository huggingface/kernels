{
  lib,
  buildPythonPackage,
  fetchFromGitHub,

  setuptools,
}:

buildPythonPackage {
  pname = "hipify-torch";
  version = "1.1.0-unstable-2026-03-05";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "hipify_torch";
    rev = "1ea3231415a41c07f9e1f1d41906df08a9af390d";
    hash = "sha256-WuPZ1hCX9B97lnd1AqDleFmRrbjuMjlS9VBiwmFNsqA=";
  };

  # Upstream does not install the v2 version of hipify. Torch uses v2 of
  # hipify, so we want to use the same to avoid divergences between Torch
  # kernels and e.g. TVM-FFI.
  postPatch = ''
    substituteInPlace setup.py \
      --replace-fail "packages=['hipify_torch',]" "packages=['hipify_torch', 'hipify_torch.v2']"
  '';

  build-system = [ setuptools ];

  pythonImportsCheck = [
    "hipify_torch.hipify_python"
    "hipify_torch.v2.hipify_python"
  ];

  meta = {
    description = "Convert CUDA C/C++ code into HIP C/C++ code";
    homepage = "https://github.com/ROCm/hipify_torch";
    license = lib.licenses.mit;
  };
}
