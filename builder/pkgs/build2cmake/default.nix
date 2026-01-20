{
  lib,
  rustPlatform,
  pkg-config,
  libgit2,
  openssl,
}:

let
  version = (builtins.fromTOML (builtins.readFile ../../build2cmake/Cargo.toml)).package.version;
in
rustPlatform.buildRustPackage {
  inherit version;
  pname = "build2cmake";

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
          "py"
          "rs"
        ]);
    in
    lib.fileset.toSource {
      root = ../../build2cmake;
      fileset = lib.fileset.fileFilter sourceFiles ../../build2cmake;
    };

  cargoLock = {
    lockFile = ../../build2cmake/Cargo.lock;
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
