{
  lib,
  rustPlatform,
  pkg-config,
  libgit2,
  openssl,
}:

let
  version = (builtins.fromTOML (builtins.readFile ../../../kernel-builder/Cargo.toml)).package.version;
in
rustPlatform.buildRustPackage {
  inherit version;
  pname = "kernel-builder";

  src =
    let
      sourceFiles =
        file:
        file.name == "Cargo.toml"
        || file.name == "Cargo.lock"
        || file.name == "pyproject.toml"
        || file.name == "pyproject_universal.toml"
        || file.name == "python_dependencies.json"
        || (builtins.any file.hasExt [
          "cmake"
          "h"
          "in"
          "py"
          "rs"
        ]);
    in
    lib.fileset.toSource {
      root = ../../../kernel-builder;
      fileset = lib.fileset.fileFilter sourceFiles ../../../kernel-builder;
    };

  cargoLock = {
    lockFile = ../../../kernel-builder/Cargo.lock;
  };

  nativeBuildInputs = [ pkg-config ];

  buildInputs = [
    libgit2
    openssl.dev
  ];

  meta = {
    description = "Create cmake build infrastructure from build.toml files";
  };
}
