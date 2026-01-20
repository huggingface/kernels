{
  lib,
  buildPythonPackage,
  rustPlatform,
}:

let
  version =
    (builtins.fromTOML (builtins.readFile ../../../kernel-abi-check/kernel-abi-check/Cargo.toml))
    .package.version;
in
buildPythonPackage {
  pname = "kernel-abi-check";
  inherit version;
  format = "pyproject";

  src =
    let
      sourceFiles =
        file:
        file.name == "Cargo.toml"
        || file.name == "Cargo.lock"
        || file.name == "manylinux-policy.json"
        || file.hasExt "pyi"
        || file.name == "pyproject.toml"
        || file.hasExt "rs"
        || file.name == "stable_abi.toml";
    in
    lib.fileset.toSource {
      root = ../../../kernel-abi-check;
      fileset = lib.fileset.fileFilter sourceFiles ../../../kernel-abi-check;
    };

  cargoDeps = rustPlatform.importCargoLock {
    lockFile = ../../../kernel-abi-check/bindings/python/Cargo.lock;
  };

  sourceRoot = "source/bindings/python";

  build-system = [
    rustPlatform.cargoSetupHook
    rustPlatform.maturinBuildHook
  ];

  meta = with lib; {
    description = "Check ABI compliance of Hugging Face Hub kernels";
  };
}
