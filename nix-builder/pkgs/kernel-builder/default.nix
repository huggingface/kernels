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
  cargoFlags = [
    "-p"
    "hf-kernel-builder"
  ];
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
        || file.name == "flake.nix"
        || file.name == "pyproject.toml"
        || file.name == "pyproject_universal.toml"
        || file.name == "python_dependencies.json"
        || file.name == ".gitattributes"
        || file.name == ".gitignore"
        || (builtins.any file.hasExt [
          "cmake"
          "cpp"
          "cu"
          "h"
          "in"
          "md"
          "metal"
          "mm"
          "py"
          "rs"
          "toml"
        ]);
    in
    with lib.fileset;
    toSource {
      root = ../../..;
      fileset = unions [
        ../../../Cargo.lock
        ../../../Cargo.toml
        (fileFilter sourceFiles ../../../kernel-builder)
        # Cargo wants access to the whole workspace.
        (fileFilter sourceFiles ../../../kernel-abi-check)
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

  # e2e tests look for binary at target/debug/ which doesn't exist in nix
  doCheck = false;

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
