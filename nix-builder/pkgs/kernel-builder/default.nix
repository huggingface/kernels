{
  lib,
  installShellFiles,
  rustPlatform,
  pkg-config,
  libgit2,
  openssl,
}:

let
  version =
    (builtins.fromTOML (builtins.readFile ../../../kernel-builder/Cargo.toml)).package.version;
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
      root = ../../..;
      fileset = lib.fileset.unions [
        (lib.fileset.fileFilter sourceFiles ../../../kernel-builder)
        (lib.fileset.fileFilter sourceFiles ../../../kernels-data)
      ];
    };

  sourceRoot = "source/kernel-builder";

  cargoLock = {
    lockFile = ../../../kernel-builder/Cargo.lock;
    outputHashes = {
      "huggingface-hub-0.1.0" = "sha256-dvYAxYj7rqNvxu5vz9LgAaODicNajnsSXhjUE0/lTjI=";
      "hf-xet-1.4.0" = "sha256-/vvU8qy9U+suiH9MCcxrV3Ayw84yRV6EmW0yzB7Uvng=";
    };
  };

  nativeBuildInputs = [
    installShellFiles
    pkg-config
  ];

  buildInputs = [
    libgit2
    openssl.dev
  ];

  postInstall = ''
    for shell in bash fish zsh; do
      $out/bin/kernel-builder completions $shell > kernel-builder.$shell
    done
    installShellCompletion kernel-builder.{bash,fish,zsh}
  '';

  meta = {
    description = "Create cmake build infrastructure from build.toml files";
  };
}
