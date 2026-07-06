{
  lib,
  installShellFiles,
  rustPlatform,
  pkg-config,
  libgit2,
  openssl,

  # Git provenance of `kernel-builder` itself. Burned into the binary at build
  # time (the build sandbox has no `.git`, so `build.rs` cannot detect it) and
  # later recorded in the build metadata of the kernels it builds. `null`/`false`
  # when `kernel-builder` is built from a non-git source (e.g. a local `path:`).
  builderRev ? null,
  builderDirty ? false,
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

  # Consumed by `kernel-builder/build.rs` and baked into the binary.
  KERNEL_BUILDER_GIT_SHA = lib.optionalString (builderRev != null) (
    lib.removeSuffix "-dirty" builderRev
  );
  KERNEL_BUILDER_GIT_DIRTY = if builderDirty then "1" else "0";

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

  setupHooks = [
    ./check-kernel-abi-hook.sh
    ./check-kernel-build-hook.sh
  ];

  meta = {
    description = "Create cmake build infrastructure from build.toml files";
  };
}
