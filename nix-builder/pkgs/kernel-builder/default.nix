{
  lib,
  installShellFiles,
  rustPlatform,
  pkg-config,
  libgit2,
  openssl,
  kernelsCargoDeps,
  mkKernelsRustSrc,
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
    ./add-provenance-hook.sh
    ./check-kernel-abi-hook.sh
    ./check-kernel-build-hook.sh
  ];

  meta = {
    description = "Create cmake build infrastructure from build.toml files";
  };
}
