{
  lib,
  rustPlatform,
}:

let
  version =
    (builtins.fromTOML (builtins.readFile ../../../kernel-abi-check/kernel-abi-check/Cargo.toml))
    .package.version;
  cargoFlags = [
    "-p"
    "kernel-abi-check"
  ];
in
rustPlatform.buildRustPackage {
  inherit version;
  pname = "kernel-abi-check";

  src =
    let
      sourceFiles =
        file:
        file.name == "Cargo.toml"
        || file.name == "Cargo.lock"
        || file.name == "manylinux-policy.json"
        || file.hasExt "rs"
        || file.name == "stable_abi.toml";
    in
    with lib.fileset;
    toSource {
      root = ../../..;
      fileset = unions [
        ../../../Cargo.lock
        ../../../Cargo.toml
        (fileFilter sourceFiles ../../../kernel-abi-check)
        # Cargo wants access to the whole workspace.
        (fileFilter sourceFiles ../../../kernel-builder)
        (fileFilter sourceFiles ../../../kernels-data)
      ];
    };

  cargoLock = {
    lockFile = ../../../Cargo.lock;
    outputHashes = {
      "huggingface-hub-0.0.1" = "sha256-By8b1NUPWu+XF3Om1NcEO+o2qdZUco+FxvrJGNRqxWs=";
    };
  };

  cargoBuildFlags = cargoFlags;
  cargoTestFlags = cargoFlags;

  setupHook = ./kernel-abi-check-hook.sh;

  meta = {
    description = "Check glibc and libstdc++ ABI compat";
  };
}
