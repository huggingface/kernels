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
    import ../../crate-dirs.nix {
      inherit lib sourceFiles;
    };

  cargoDeps = rustPlatform.importCargoLock {
    lockFile = ../../../../Cargo.lock;
    outputHashes = {
      "hf-hub-1.0.0" = "sha256-XJVbG/dfxeSaTvyZMqB/6oF0I5cqKXIXzG5Zq00xmnk=";
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
