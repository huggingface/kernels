{
  lib,
  cargo,
  rustc,
  rustPlatform,
}:

{ extension, src }:
let
  lockFile = src + "/Cargo.lock";
in
assert lib.assertMsg (builtins.pathExists lockFile) ''
  Rust kernels require a `Cargo.lock` in the project root, listed in the
  component's `src` in build.toml so that it reaches the build.'';

extension.overrideAttrs (previous: {
  cargoDeps = rustPlatform.importCargoLock {
    inherit lockFile;
    allowBuiltinFetchGit = true;
  };
  nativeBuildInputs = previous.nativeBuildInputs ++ [
    rustPlatform.cargoSetupHook
    cargo
    rustc
  ];
})
