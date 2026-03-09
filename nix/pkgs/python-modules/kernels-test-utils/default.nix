{
  lib,
  buildPythonPackage,
  setuptools,

  pytest,
  torch,
}:

let
  version =
    (builtins.fromTOML (builtins.readFile ../../../../kernels-test-utils/pyproject.toml)).project.version;
in
buildPythonPackage {
  pname = "kernels-test-utils";
  inherit version;
  pyproject = true;

  src =
    let
      sourceFiles = file: file.hasExt "toml" || file.hasExt "py";
    in
    lib.fileset.toSource {
      root = ../../../../kernels-test-utils;
      fileset = lib.fileset.fileFilter sourceFiles ../../../../kernels-test-utils;
    };

  build-system = [ setuptools ];

  dependencies = [
    pytest
    torch
  ];

  pythonImportsCheck = [
    "kernels_test_utils"
  ];

  meta = with lib; {
    description = "Shared test utilities for kernel repos";
  };
}
