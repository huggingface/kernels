{
  lib,
  installShellFiles,
  rustPlatform,
  pkg-config,
  libgit2,
  openssl,
  kernelsCargoDeps,
  mkKernelsRustSrc,

  # Git provenance (`{ sha, dirty }`, or `null` for a non-git source) of the
  # `kernel-builder` flake. It is burned into the binary at build time and
  # later recorded in the build metadata of the kernels it builds. The build
  # sandbox has no `.git`, so `build.rs` cannot detect it.
  builderProvenance ? null,
}:

let
  version =
    (builtins.fromTOML (builtins.readFile ../../../kernel-builder/Cargo.toml)).package.version;
  cargoFlags = [
    "-p"
    "hf-kernel-builder"
  ];
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

    src = mkKernelsRustSrc {
      crates = [ "kernel-builder" ];
    };

    cargoDeps = kernelsCargoDeps;

    cargoBuildFlags = cargoFlags;

    # Only run the unit tests in `src/` (`--bins`). e2e tests in `tests/`
    # (which do not work in the build sandbox) are not run.
    cargoTestFlags = cargoFlags ++ [ "--bins" ];

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
