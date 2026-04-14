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
    lib.fileset.toSource {
      root = ../../..;
      fileset = lib.fileset.fileFilter sourceFiles ../../..;
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
