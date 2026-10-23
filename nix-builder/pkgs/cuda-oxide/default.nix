# cuda-oxide is a rustc codegen plugin and must use its matching Rust toolchain.
{
  fetchFromGitHub,
  libffi,
  libxml2,
  makeRustPlatform,
  rust-bin,
  zlib,
  zstd,
}:

let
  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "cuda-rust";
    rev = "6d47ae49cb256a07948fb499c20c3eb9b2a96268";
    hash = "sha256-g9qg2zHLybO8kPLoOJqnYbWjRYtjGV3QVT+X4f8+1fg=";
  };

  # rust-overlay propagates the compiler of its own package set. Drop it so
  # kernels link with their (manylinux) stdenv, as with the rust-cpu toolchain.
  toolchain =
    (rust-bin.fromRustupToolchainFile "${src}/cuda-oxide/rust-toolchain.toml").overrideAttrs
      {
        propagatedBuildInputs = [ ];
        depsHostHostPropagated = [ ];
      };

  rustPlatform = makeRustPlatform {
    cargo = toolchain;
    rustc = toolchain;
  };
in
{
  inherit toolchain;

  backend = rustPlatform.buildRustPackage {
    pname = "rustc-codegen-cuda";
    version = "0.2.1";
    inherit src;
    sourceRoot = "source/cuda-oxide/crates/rustc-codegen-cuda";
    cargoLock = {
      lockFile = "${src}/cuda-oxide/crates/rustc-codegen-cuda/Cargo.lock";
      outputHashes."pliron-0.18.0" = "sha256-HQ/xNfQcIf9pQRrgxCYGFdVfdCMaE4iwj/G9mgAJOWw=";
    };
    buildInputs = [
      libffi
      libxml2
      zlib
      zstd
    ];
    doCheck = false;
  };
}
