# cuda-oxide is a rustc codegen plugin and must use its matching Rust toolchain.
{
  pkgs,
  lib,
  stdenv,
  cudaPackages,
}:
let
  src = pkgs.fetchFromGitHub {
    owner = "NVIDIA";
    repo = "cuda-rust";
    rev = "6d47ae49cb256a07948fb499c20c3eb9b2a96268";
    hash = "sha256-g9qg2zHLybO8kPLoOJqnYbWjRYtjGV3QVT+X4f8+1fg=";
  };
  toolchain = pkgs.rust-bin.fromRustupToolchainFile "${src}/cuda-oxide/rust-toolchain.toml";
  platform = pkgs.makeRustPlatform {
    cargo = toolchain;
    rustc = toolchain;
  };
  backend = platform.buildRustPackage {
    pname = "rustc-codegen-cuda";
    version = "0.2.1";
    inherit src;
    sourceRoot = "source/cuda-oxide/crates/rustc-codegen-cuda";
    cargoLock = {
      lockFile = "${src}/cuda-oxide/crates/rustc-codegen-cuda/Cargo.lock";
      outputHashes."pliron-0.18.0" = "sha256-HQ/xNfQcIf9pQRrgxCYGFdVfdCMaE4iwj/G9mgAJOWw=";
    };
    buildInputs = with pkgs; [
      libffi
      libxml2
      zstd
      zlib
    ];
    doCheck = false;
  };
  cudaHome = pkgs.symlinkJoin {
    name = "cuda-home";
    paths = with cudaPackages; [
      (lib.getDev cuda_cudart)
      (lib.getLib cuda_cudart)
      (lib.getDev cccl)
      (lib.getOutput "include" libcurand)
      cuda_crt
    ];
  };
in
{
  inherit toolchain backend;
  env = {
    CUDA_HOME = cudaHome;
    CUDA_TOOLKIT_PATH = cudaHome;
    LIBRARY_PATH = "${lib.getOutput "stubs" cudaPackages.cuda_cudart}/lib/stubs";
    LIBCLANG_PATH = "${lib.getLib pkgs.libclang}/lib";
    BINDGEN_EXTRA_CLANG_ARGS = lib.concatStringsSep " " [
      "-isystem ${lib.getLib pkgs.libclang}/lib/clang/${lib.versions.major pkgs.libclang.version}/include"
      "-isystem ${lib.getDev stdenv.cc.libc}/include"
    ];
  };
}
