{
  lib,
  rustPlatform,
}:

let
  version = (builtins.fromTOML (builtins.readFile ../../../kernel-port/Cargo.toml)).package.version;
  cargoFlags = [
    "-p"
    "kernel-port"
  ];
in
rustPlatform.buildRustPackage {
  inherit version;
  pname = "kernel-port";

  src =
    let
      sourceFiles =
        file:
        file.name == "Cargo.toml"
        || file.name == "Cargo.lock"
        || file.name == "flake.nix"
        || file.name == "manylinux-policy.json"
        || file.name == "pyproject.toml"
        || file.name == "pyproject_universal.toml"
        || file.name == "python_dependencies.json"
        || file.name == "shim_function_versions.txt"
        || file.name == "stable_abi.toml"
        || file.name == ".gitattributes"
        || file.name == ".gitignore"
        || (builtins.any file.hasExt [
          "cmake"
          "cpp"
          "cu"
          "h"
          "in"
          "kdl"
          "md"
          "metal"
          "mm"
          "py"
          "rs"
          "toml"
        ]);
    in
    import ../crate-dirs.nix {
      inherit lib sourceFiles;
    };

  cargoLock = {
    lockFile = ../../../Cargo.lock;
    outputHashes = {
      "hf-hub-1.1.0" = "sha256-wClUTCmphrO4QM+IYwYrNxyvDp8qBGAPdP+Wca8TgRA=";
    };
  };

  cargoBuildFlags = cargoFlags;
  cargoTestFlags = cargoFlags;

  meta = {
    description = "Port third-party kernels to the Hugging Face Kernels layout";
    mainProgram = "kernel-port";
  };
}
