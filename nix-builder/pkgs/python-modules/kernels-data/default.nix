{
  lib,
  buildPythonPackage,
  rustPlatform,
}:

let
  version =
    (builtins.fromTOML (builtins.readFile ../../../../kernels-data/Cargo.toml)).package.version;
  cargoFlags = [
    "-m"
    "kernels-data/bindings/python/Cargo.toml"
  ];
in
buildPythonPackage {
  pname = "kernels-data";
  inherit version;
  format = "pyproject";

  src =
    let
      sourceFiles =
        file:
        file.name == "Cargo.toml"
        || file.name == "Cargo.lock"
        || file.hasExt "pyi"
        || file.name == "pyproject.toml"
        || file.name == "python_dependencies.json"
        || file.hasExt "rs";
    in
    import ../../crate-dirs.nix {
      inherit lib sourceFiles;
    };

  cargoDeps = rustPlatform.importCargoLock {
    lockFile = ../../../../Cargo.lock;
    outputHashes = {
      "huggingface-hub-0.0.1" = "sha256-By8b1NUPWu+XF3Om1NcEO+o2qdZUco+FxvrJGNRqxWs=";
    };
  };

  maturinBuildFlags = cargoFlags;

  build-system = [
    rustPlatform.cargoSetupHook
    rustPlatform.maturinBuildHook
  ];

  meta = with lib; {
    description = "Kernels data structures";
  };
}
