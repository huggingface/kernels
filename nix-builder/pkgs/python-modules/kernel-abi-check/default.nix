{
  lib,
  buildPythonPackage,
  rustPlatform,
}:

let
  version =
    (builtins.fromTOML (builtins.readFile ../../../../kernel-abi-check/kernel-abi-check/Cargo.toml))
    .package.version;
  cargoFlags = [
    "-m"
    "kernel-abi-check/bindings/python/Cargo.toml"
  ];
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
    with lib.fileset;
    toSource {
      root = ../../../..;
      fileset = unions [
        ../../../../Cargo.lock
        ../../../../Cargo.toml
        (fileFilter sourceFiles ../../../../kernel-abi-check)
        # Cargo wants access to the whole workspace.
        (fileFilter sourceFiles ../../../../kernel-builder)
        (fileFilter sourceFiles ../../../../kernels-data)
      ];
    };

  cargoDeps = rustPlatform.importCargoLock {
    lockFile = ../../../../Cargo.lock;
    outputHashes = {
      "huggingface-hub-0.0.1" = "sha256-By8b1NUPWu+XF3Om1NcEO+o2qdZUco+FxvrJGNRqxWs=";
    };
  };

  maturinBuildFlags = cargoFlags;

  #sourceRoot = "source/bindings/python";

  build-system = [
    rustPlatform.cargoSetupHook
    rustPlatform.maturinBuildHook
  ];

  meta = with lib; {
    description = "Check ABI compliance of Hugging Face Hub kernels";
  };
}
