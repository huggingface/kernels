final: prev:
let
  # For XPU we use MKL from the joined oneAPI toolkit.
  useMKL = final.stdenv.isx86_64 && !(final.config.xpuSupport or false);
in
{
  # Use MKL for BLAS/LAPACK on x86_64.
  blas = if useMKL then prev.blas.override { blasProvider = prev.mkl; } else prev.blas;
  lapack = if useMKL then prev.lapack.override { lapackProvider = prev.mkl; } else prev.blas;

  kernel-builder = prev.callPackage ./pkgs/kernel-builder { };

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
      python-self: python-super:
      with python-self;
      let
        triton-xpu = callPackage ./pkgs/python-modules/triton-xpu { };
      in
      {
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

        huggingface-hub = python-super.huggingface-hub.overridePythonAttrs (prevAttrs: rec {
          version = "1.8.0";
          src = python-super.fetchPypi {
            pname = "huggingface_hub";
            inherit version;
            hash = "sha256-xWJ7L9Uh4Ayvjv9KyWW6mI6nUWf61+5y4X+bcYPsY/M=";
          };
          dependencies =
            (prevAttrs.dependencies or [ ])
            ++ (with python-self; [
              httpx
              shellingham
              typer
            ]);
          # Skip tests since they require network access.
          doCheck = false;
        });

        fastapi = python-super.fastapi.overrideAttrs (
          _: prevAttrs: {
            # Gets stuck sometimes, already tested in nixpkgs.
            doInstallCheck = false;
          }
        );

        hf-xet = python-super.hf-xet.overridePythonAttrs (prevAttrs: rec {
          version = "1.4.2";
          src = final.fetchFromGitHub {
            owner = "huggingface";
            repo = "xet-core";
            tag = "v${version}";
            hash = "sha256-UdHEpJztlVI8LPs8Ne9sKe1Nv3kVVk4YLxQ3W8sUPbQ=";
          };
          cargoDeps = final.rustPlatform.fetchCargoVendor {
            inherit (prevAttrs)
              pname
              sourceRoot
              ;
            inherit
              version
              src
              ;
            hash = "sha256-GV+XY5uV57yQWVGdRLpGU3eD8Gz2gy6p7OHlF+mlJI4=";
          };
        });

        nvidia-cutlass-dsl = python-self.callPackage ./pkgs/python-modules/nvidia-cutlass-dsl { };

        nvidia-cutlass-dsl-libs = python-self.callPackage ./pkgs/python-modules/nvidia-cutlass-dsl-libs { };

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

        # Remove once sglang moves to a newer Torch version.
        torch-bin_2_9 = mkTorch {
          version = "2.9";
          triton-xpu = null;
          # Not supported anymore.
          xpuPackages = null;
        };

        torch-bin_2_10 = mkTorch {
          version = "2.10";
          triton-xpu = triton-xpu_3_6_0;
          xpuPackages = final.xpuPackages_2025_3_1;
        };

        torch-bin_2_11 = mkTorch {
          version = "2.11";
          triton-xpu = triton-xpu_3_7_0;
          xpuPackages = final.xpuPackages_2025_3_2;
        };

        transformers = python-super.transformers.overridePythonAttrs (prevAttrs: rec {
          version = "5.3.0";
          src = python-super.fetchPypi {
            pname = "transformers";
            inherit version;
            hash = "sha256-AJVVs2QCnanilG1B8cXenxXmsd9GsYm3KT8zoWG5xVc=";
          };

          dependencies = (prevAttrs.dependencies or [ ]) ++ [
            python-self.typer
          ];
        });

        triton-xpu_3_6_0 = triton-xpu.triton-xpu_3_6_0;

        triton-xpu_3_7_0 = triton-xpu.triton-xpu_3_7_0;

        tvm-ffi = callPackage ./pkgs/python-modules/tvm-ffi {
        };
      }
    )
    (import ./pkgs/python-modules/hooks)
  ];

  xpuPackages = final.xpuPackages_2025_3_1;
}
// (import ./pkgs/cutlass { pkgs = final; })
// (
  let
    flattenVersion = prev.lib.strings.replaceStrings [ "." ] [ "_" ];
    readPackageMetadata = path: (builtins.fromJSON (builtins.readFile path));
    versions = [
      "7.0.2"
      "7.1.1"
      "7.2.1"
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
      "2025.3.1"
      "2025.3.2"
    ];
    newXpuPackages = final.callPackage ./pkgs/xpu-packages { };
  in
  builtins.listToAttrs (
    map (version: {
      name = "xpuPackages_${flattenVersion version}";
      value = newXpuPackages {
        packageMetadata = readPackageMetadata ./pkgs/xpu-packages/intel-deep-learning-${version}.json;
      };
    }) xpuVersions
  )
)
