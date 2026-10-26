{
  lib,
  buildPythonPackage,
  pytestCheckHook,
  setuptools,

  kernels,
}:

let
  version = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project.version;
in
buildPythonPackage {
  pname = "generate-symbols";
  inherit version;
  pyproject = true;

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./pyproject.toml
      ./src
      ./tests
    ];
  };

  build-system = [ setuptools ];

  dependencies = [ kernels ];

  nativeCheckInputs = [ pytestCheckHook ];

  pythonImportsCheck = [ "generate_symbols" ];

  meta = {
    description = "Generate public API symbols for built kernels";
    license = lib.licenses.asl20;
    mainProgram = "generate-symbols";
  };
}
