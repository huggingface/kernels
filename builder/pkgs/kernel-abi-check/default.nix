{
  lib,
  rustPlatform,
}:

let
  version =
    (builtins.fromTOML (builtins.readFile ../../../kernel-abi-check/kernel-abi-check/Cargo.toml))
    .package.version;
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
      root = ../../../kernel-abi-check/kernel-abi-check;
      fileset = lib.fileset.fileFilter sourceFiles ../../../kernel-abi-check/kernel-abi-check;
    };

  cargoLock = {
    lockFile = ../../../kernel-abi-check/kernel-abi-check/Cargo.lock;
  };

  setupHook = ./kernel-abi-check-hook.sh;

  meta = {
    description = "Check glibc and libstdc++ ABI compat";
  };
}
