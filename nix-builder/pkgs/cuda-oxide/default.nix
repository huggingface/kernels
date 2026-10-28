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
  # NVIDIA/cuda-rust main with `CudaStream::borrow_raw`, which kernels need to
  # launch on the Torch stream. Move back to NVIDIA/cuda-rust once
  # https://github.com/NVIDIA/cuda-rust/pull/1601 is merged.
  src = fetchFromGitHub {
    owner = "drbh";
    repo = "cuda-oxide";
    rev = "d9d4e379e9a80d1ce7c681d88a086aacec95ba23";
    hash = "sha256-/rfi6TODuZp/YSoAy/bpse3nYJSokpmMwg3bHTtCNBM=";
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
    cargoLock.lockFile = "${src}/cuda-oxide/crates/rustc-codegen-cuda/Cargo.lock";
    buildInputs = [
      libffi
      libxml2
      zlib
      zstd
    ];
    doCheck = false;
  };
}
