{
  lib,
  buildPythonPackage,
  setuptools,
  pytestCheckHook,
  kernels,
}:

buildPythonPackage {
  pname = "generate-symbols";
  version = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project.version;
  pyproject = true;

  src = ./.;

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
