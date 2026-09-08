{
  lib,
  buildPythonPackage,
  rustPlatform,

  huggingface-hub,
  pyyaml,
  sigstore,
  tomlkit,
  torch,
}:

let
  version =
    (builtins.fromTOML (builtins.readFile ../../../../kernels/pyproject.toml)).project.version;
  cargoFlags = [
    "-m"
    "kernels/Cargo.toml"
  ];
in
buildPythonPackage {
  pname = "kernels";
  inherit version;
  format = "pyproject";

  src =
    let
      sourceFiles =
        file:
        file.name == "README.md"
        || file.name == "Cargo.toml"
        || file.name == "Cargo.lock"
        || file.hasExt "rs"
        || file.hasExt "pyi"
        || file.hasExt "lock"
        || file.hasExt "json"
        || file.hasExt "toml"
        || file.hasExt "py";
    in
    import ../../crate-dirs.nix {
      inherit lib sourceFiles;
    };

  cargoDeps = rustPlatform.importCargoLock {
    lockFile = ../../../../Cargo.lock;
    outputHashes = {
      "hf-hub-1.1.0" = "sha256-wClUTCmphrO4QM+IYwYrNxyvDp8qBGAPdP+Wca8TgRA=";
    };
  };

  maturinBuildFlags = cargoFlags;

  build-system = [
    rustPlatform.cargoSetupHook
    rustPlatform.maturinBuildHook
  ];

  dependencies = [
    huggingface-hub
    pyyaml
    sigstore
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
