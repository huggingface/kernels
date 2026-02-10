{
  lib,
  buildPythonPackage,
  setuptools,

  huggingface-hub,
  kernel-abi-check,
  pyyaml,
  tomlkit,
  torch,
}:

let
  version =
    (builtins.fromTOML (builtins.readFile ../../../../kernels/pyproject.toml)).project.version;
in
buildPythonPackage {
  pname = "kernels";
  inherit version;
  pyproject = true;

  src =
    let
      sourceFiles =
        file: file.hasExt "lock" || file.hasExt "json" || file.hasExt "toml" || file.hasExt "py";
    in
    lib.fileset.toSource {
      root = ../../../../kernels;
      fileset = lib.fileset.fileFilter sourceFiles ../../../../kernels;
    };

  build-system = [ setuptools ];

  dependencies = [
    huggingface-hub
    kernel-abi-check
    pyyaml
    tomlkit
    torch
  ];

  pythonImportsCheck = [
    "kernels"
  ];

  meta = with lib; {
    description = "Python client for the Kernel Hub";
  };
}
