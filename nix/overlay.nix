final: prev:
let
  # For XPU we use MKL from the joined oneAPI toolkit.
  useMKL = final.stdenv.isx86_64 && !(final.config.xpuSupport or false);
in
{
  # Use MKL for BLAS/LAPACK on x86_64.
  blas = if useMKL then prev.blas.override { blasProvider = prev.mkl; } else prev.blas;
  lapack = if useMKL then prev.lapack.override { lapackProvider = prev.mkl; } else prev.blas;

  build2cmake = prev.callPackage ./pkgs/build2cmake { };

  cmakeNvccThreadsHook = prev.callPackage ./pkgs/cmake-nvcc-threads-hook { };

  get-kernel-check = prev.callPackage ./pkgs/get-kernel-check { };

  kernel-abi-check = prev.callPackage ./pkgs/kernel-abi-check { };

  kernel-layout-check = prev.callPackage ./pkgs/kernel-layout-check { };

  # Used by ROCm.
  libffi_3_2 = final.libffi_3_3.overrideAttrs (
    finalAttrs: _: {
      version = "3.2.1";
      src = final.fetchurl {
        url = with finalAttrs; "https://gcc.gnu.org/pub/${pname}/${pname}-${version}.tar.gz";
        hash = "sha256-0G67jh2aItGeONY/24OVQlPzm+3F1GIyoFZFaFciyjc=";
      };
    }
  );

  nvtx = final.callPackage ./pkgs/nvtx { };

  metal-cpp = final.callPackage ./pkgs/metal-cpp { };

  rewrite-nix-paths-macho = prev.callPackage ./pkgs/rewrite-nix-paths-macho { };

  remove-bytecode-hook = prev.callPackage ./pkgs/remove-bytecode-hook { };

  stdenvGlibc_2_27 = import ./pkgs/stdenv-glibc-2_27 {
    # Do not use callPackage, because we want overrides to apply to
    # the stdenv itself and not this file.
    inherit (final)
      config
      fetchFromGitHub
      overrideCC
      wrapBintoolsWith
      wrapCCWith
      gcc13Stdenv
      stdenv
      bintools-unwrapped
      cudaPackages
      libgcc
      ;
  };

  ucx = prev.ucx.overrideAttrs (
    _: prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [ final.cudaPackages.cuda_nvcc ];
    }
  );

  # Python packages
  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
    (
      python-self: python-super: with python-self; {
        cuda-bindings = python-self.callPackage ./pkgs/python-modules/cuda-bindings { };

        cuda-pathfinder = python-self.callPackage ./pkgs/python-modules/cuda-pathfinder { };

        # Starting with the CUDA 12.8 version, cuda-python is a metapackage
        # that pulls in relevant dependencies. For CUDA 12.6 it is just
        # cuda-bindings.
        cuda-python =
          if final.cudaPackages.cudaMajorMinorVersion == "12.6" then
            python-self.cuda-bindings
          else
            python-self.callPackage ./pkgs/python-modules/cuda-python { };

        nvidia-cutlass-dsl = python-self.callPackage ./pkgs/python-modules/nvidia-cutlass-dsl { };

        kernel-abi-check = callPackage ./pkgs/python-modules/kernel-abi-check { };

        kernels = callPackage ./pkgs/python-modules/kernels { };

        pyclibrary = python-self.callPackage ./pkgs/python-modules/pyclibrary { };

        mkTorch = callPackage ./pkgs/python-modules/torch/binary { };

        scipy = python-super.scipy.overrideAttrs (
          _: prevAttrs: {
            # Three tests have a slight deviance.
            doCheck = false;
            doInstallCheck = false;
          }
        );

        torch-bin_2_9 = mkTorch {
          version = "2.9";
          xpuPackages = final.xpuPackages_2025_2;
        };

        torch-bin_2_10 = mkTorch {
          version = "2.10";
          xpuPackages = final.xpuPackages_2025_3;
        };

        triton-xpu_2_9 = callPackage ./pkgs/python-modules/triton-xpu {
          torchVersion = "2.9";
          xpuPackages = final.xpuPackages_2025_2;
        };
      }
    )
    (import ./pkgs/python-modules/hooks)
  ];

  xpuPackages = final.xpuPackages_2025_1;
}
// (import ./pkgs/cutlass { pkgs = final; })
// (
  let
    flattenVersion = prev.lib.strings.replaceStrings [ "." ] [ "_" ];
    readPackageMetadata = path: (builtins.fromJSON (builtins.readFile path));
    versions = [
      "6.3.4"
      "6.4.2"
      "7.0.2"
      "7.1.1"
    ];
    newRocmPackages = final.callPackage ./pkgs/rocm-packages { };
  in
  builtins.listToAttrs (
    map (version: {
      name = "rocmPackages_${flattenVersion (prev.lib.versions.majorMinor version)}";
      value = newRocmPackages {
        packageMetadata = readPackageMetadata ./pkgs/rocm-packages/rocm-${version}-metadata.json;
      };
    }) versions
  )
)
// (
  let
    flattenVersion = prev.lib.strings.replaceStrings [ "." ] [ "_" ];
    readPackageMetadata = path: (builtins.fromJSON (builtins.readFile path));
    xpuVersions = [
      "2025.1.3"
      "2025.2.1"
      "2025.3.1"
    ];
    newXpuPackages = final.callPackage ./pkgs/xpu-packages { };
  in
  builtins.listToAttrs (
    map (version: {
      name = "xpuPackages_${flattenVersion (prev.lib.versions.majorMinor version)}";
      value = newXpuPackages {
        packageMetadata = readPackageMetadata ./pkgs/xpu-packages/intel-deep-learning-${version}.json;
      };
    }) xpuVersions
  )
)
