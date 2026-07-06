{
  lib,
  installShellFiles,
  rustPlatform,
  pkg-config,
  libgit2,
  openssl,

  # The `kernel-builder` flake itself (or `null` for a non-git source). Its git
  # provenance is burned into the binary at build time and later recorded in the
  # build metadata of the kernels it builds. The build sandbox has no `.git`, so
  # `build.rs` cannot detect it; it is derived from the flake here instead.
  builderSelf ? null,
}:

let
  version =
    (builtins.fromTOML (builtins.readFile ../../../kernel-builder/Cargo.toml)).package.version;
  cargoFlags = [
    "-p"
    "hf-kernel-builder"
  ];

  builderProvenance = import ../../lib/flake-provenance.nix { inherit lib; } builderSelf;
in
rustPlatform.buildRustPackage (
  # Supply the git provenance through `built`'s override environment variables
  # (`hf_kernel_builder` is the package name with hyphens replaced by
  # underscores), which `build.rs` bakes into the binary. When there is no provenance
  # information (e.g. non-git source), do not set the variables.
  lib.optionalAttrs (builderProvenance != null) {
    "BUILT_OVERRIDE_hf_kernel_builder_GIT_COMMIT_HASH" = builderProvenance.sha;
    "BUILT_OVERRIDE_hf_kernel_builder_GIT_DIRTY" = if builderProvenance.dirty then "true" else "false";
  }
  // {
    inherit version;
    pname = "kernel-builder";

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
)
