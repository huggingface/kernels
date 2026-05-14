{
  cudaSupport ? torch.cudaSupport,
  rocmSupport ? torch.rocmSupport,
  xpuSupport ? torch.xpuSupport,

  pkgs,
  lib,
  callPackage,
  manylinux_2_28,
  stdenv,
  stdenvGlibc_2_27,
  cudaPackages,
  rocmPackages,
  writeScriptBin,
  xpuPackages,

  torch,
}:

let
  cudaBackendStdenv = import ../../pkgs/cuda/backendStdenv {
    inherit (pkgs)
      _cuda
      config
      lib
      stdenvAdapters
      ;
    inherit (cudaPackages) cudaMajorMinorVersion;
    # Allow auto-selection to use manylinux gcc versions.
    pkgs = manylinux_2_28;
    stdenv = manylinux_2_28.stdenv;
  };
  effectiveStdenv =
    if stdenv.hostPlatform.isLinux && cudaSupport then
      cudaBackendStdenv
    else if stdenv.hostPlatform.isLinux then
      manylinux_2_28.stdenv
    else
      stdenv;

  # CLR that uses the provided stdenv, which can be different from the default
  # to support old glibc/libstdc++ versions.
  clr = (
    rocmPackages.clr.override {
      clang = rocmPackages.llvm.clang.override {
        stdenv = effectiveStdenv;
        bintools = rocmPackages.llvm.bintools.override { libc = effectiveStdenv.cc.libc; };
        glibc = effectiveStdenv.cc.libc;
      };
    }
  );

  cuda_nvcc = cudaPackages.cuda_nvcc.override {
    backendStdenv = cudaBackendStdenv;
  };

  oneapi-torch-dev = xpuPackages.oneapi-torch-dev.override { stdenv = effectiveStdenv; };
  onednn-xpu = xpuPackages.onednn-xpu.override {
    inherit oneapi-torch-dev;
    stdenv = effectiveStdenv;
  };
in
{
  extraBuildDeps =
    lib.optionals xpuSupport [
      oneapi-torch-dev
      onednn-xpu
    ]
    ++ lib.optionals rocmSupport (
      [
        clr
      ]
      ++ (with rocmPackages; [
        hipcub-devel
        hipsparselt
        rocprim-devel
        rocthrust-devel
        rocwmma-devel
      ])
    );

  mkTvmFfiExtension = callPackage ./tvm-ffi/arch.nix {
    inherit
      clr
      cuda_nvcc
      oneapi-torch-dev
      onednn-xpu
      torch
      ;
    stdenv = effectiveStdenv;
  };

  mkTorchExtension = callPackage ./torch/arch.nix {
    inherit
      clr
      cuda_nvcc
      oneapi-torch-dev
      onednn-xpu
      torch
      ;
    stdenv = effectiveStdenv;
  };

  mkTorchNoArchExtension = callPackage ./torch/no-arch.nix { inherit torch; };

  stdenv = effectiveStdenv;
}
