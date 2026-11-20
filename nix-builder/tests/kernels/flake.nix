{
  description = "Kernels that test the kernel builder";

  inputs = {
    kernel-builder.url = "path:../../..";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    let
      inherit (kernel-builder.inputs) flake-utils nixpkgs;

      # Helpers shared with the example kernels. See that file for the
      # attributes that CI kernels support.
      ciKernelsLib = import ../ci-kernels.nix {
        inherit self kernel-builder;
      };

      inherit (ciKernelsLib)
        cudaVersion
        cuda13Version
        rocmVersion
        xpuVersion
        torchVersion
        tvmFfiVersion
        reluZipSymbols
        mkKernelOutputs
        mkKernelOutputs'
        mkCiBuild
        ;

      # All test kernels to build in CI.
      ciKernels = [
        {
          name = "cpp20-symbols-kernel";
          path = ./cpp20-symbols;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cpu-${sys}"};
        }
        {
          # Check arch intersection. 5.0 is dropped because it is not supported.
          name = "relu-archs-subset";
          path = ./relu-archs-subset;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${cudaVersion}-${sys}"};
          checkCudaCapabilities = [
            "7.5"
            "8.0"
            "9.0"
          ];
        }
        {
          name = "relu-zip-kernel";
          path = ./relu-zip;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${cudaVersion}-${sys}"};
          checkSymbols = reluZipSymbols;
        }
        {
          name = "relu-tvm-ffi-compiler-flags-kernel";
          path = ./relu-tvm-ffi-compiler-flags;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"tvm-ffi${tvmFfiVersion}-${cudaVersion}-${sys}"};
        }
        {
          name = "cutlass-gemm-tvm-ffi-kernel";
          path = ./cutlass-gemm-tvm-ffi;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"tvm-ffi${tvmFfiVersion}-${cudaVersion}-${sys}"};
        }
        {
          # Tests that we can build with the extra torchVersions argument.
          name = "relu-specific-torch";
          path = ./relu-specific-torch;
          drv = sys: out: out.packages.${sys}.default;
          torchVersions = _defaultVersions: [
            {
              torchVersion = "2.13";
              cudaVersion = "12.6";
              system = "x86_64-linux";
              bundleBuild = true;
            }
            {
              torchVersion = "2.13";
              cudaVersion = "12.6";
              system = "aarch64-linux";
              bundleBuild = true;
            }
          ];
        }
        {
          name = "relu-compiler-flags";
          path = ./relu-compiler-flags;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${cudaVersion}-${sys}"};
        }
        {
          name = "relu-invalid-capability";
          path = ./relu-invalid-capability;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${cudaVersion}-${sys}"};
          assertFail = true;
          assertFailLogs = [ "empty set of capabilities" ];
        }
        {
          name = "kernel-deps-kernel";
          path = ./kernel-deps;
          drv = sys: out: out.packages.${sys}.redistributable.torch-cuda;
        }
        # NCCL is a Torch dependency, so its headers must be available to
        # kernels and kernels that use NCCL must link against it. NCCL differs
        # per CUDA major version, so check CUDA 12 and CUDA 13.
        {
          name = "nccl-version-kernel";
          path = ./nccl-version;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${cudaVersion}-${sys}"};
          checkSymbols = {
            functions = [
              "nccl_header_version"
              "nccl_runtime_version"
            ];
            layers = [ ];
          };
          checkNeeded = [ "libnccl.so.2" ];
        }
        {
          name = "nccl-version-kernel-cu13";
          path = ./nccl-version;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${cuda13Version}-${sys}"};
          checkNeeded = [ "libnccl.so.2" ];
        }
      ];

      # ROCm kernels to build in CI.
      ciRocmKernels = [
        {
          name = "relu-invalid-capability";
          path = ./relu-invalid-capability;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${rocmVersion}-${sys}"};
          assertFail = true;
          assertFailLogs = [ "empty set of architectures" ];
        }
        {
          # Check arch intersection. gfx940 is dropped because it is not supported.
          name = "relu-archs-subset";
          path = ./relu-archs-subset;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${rocmVersion}-${sys}"};
          checkRocmArchs = [
            "gfx90a"
            "gfx942"
            "gfx1100"
          ];
        }
        {
          name = "relu-compiler-flags";
          path = ./relu-compiler-flags;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${rocmVersion}-${sys}"};
        }
      ];

      # XPU kernels to build in CI.
      ciXpuKernels = [
        {
          name = "relu-tvm-ffi-compiler-flags-kernel";
          path = ./relu-tvm-ffi-compiler-flags;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"tvm-ffi${tvmFfiVersion}-${xpuVersion}-${sys}"};
        }
        {
          name = "relu-compiler-flags";
          path = ./relu-compiler-flags;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${xpuVersion}-${sys}"};
        }
      ];

      # Metal kernels to build in CI.
      ciMetalKernels = [
        {
          name = "bfl-mlx-kernel";
          path = ./bfl-mlx;
          drv = sys: out: out.packages.${sys}.redistributable."mlx-metal";
        }
        {
          name = "bfl-torch-kernel";
          path = ./bfl-torch;
          drv = sys: out: out.packages.${sys}.redistributable.torch-metal;
        }
      ];

      # CPU kernels to build in CI.
      ciCpuKernels = [
        {
          name = "relu-zip-kernel-cpu";
          path = ./relu-zip;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cpu-${sys}"};
          checkSymbols = reluZipSymbols;
        }
        {
          # Run the tests of the ZIP build layout kernel directly during the
          # build. This checks that `kernels` can load kernels from ZIP archives.
          name = "relu-zip-pytest";
          path = ./relu-zip;
          drv =
            sys: out:
            let
              variant = "torch${torchVersion}-cpu-${sys}";
              reluZip = out.packages.${sys}.redistributable.${variant};
              kernelPkgs = out.packages.${sys}.pkgs.${variant};
              testPython = kernelPkgs.python3.withPackages (
                ps: with ps; [
                  torch
                  pytest
                  kernels
                ]
              );
            in
            kernelPkgs.runCommand "relu-zip-pytest"
              {
                nativeBuildInputs = [ testPython ];
              }
              ''
                export HOME=$TMPDIR
                export LOCAL_KERNELS="kernels-test/relu-zip=${reluZip}"
                python -m pytest ${reluZip.src}/tests -m kernels_ci -p no:cacheprovider
                touch $out
              '';
        }
        {
          # This test only requires a CPU, so let's run the test directly during the build.
          name = "symbol-conflicts-pytest";
          path = ./symbol-conflicts2;
          drv =
            sys: _out:
            let
              variant = "torch${torchVersion}-cpu-${sys}";
              conflictsFlake = mkKernelOutputs { path = ./symbol-conflicts; };
              conflicts2Flake = mkKernelOutputs { path = ./symbol-conflicts2; };
              conflicts = conflictsFlake.packages.${sys}.redistributable.${variant};
              conflicts2 = conflicts2Flake.packages.${sys}.redistributable.${variant};
              kernelPkgs = conflictsFlake.packages.${sys}.pkgs.${variant};
              testPython = kernelPkgs.python3.withPackages (
                ps: with ps; [
                  torch
                  pytest
                  kernels
                ]
              );
            in
            kernelPkgs.runCommand "symbol-conflicts-pytest"
              {
                nativeBuildInputs = [ testPython ];
              }
              ''
                export HOME=$TMPDIR
                export LOCAL_KERNELS="kernels-test/symbol-conflicts=${conflicts}:kernels-test/symbol-conflicts2=${conflicts2}"
                python -m pytest ${conflicts2.src}/tests -m kernels_ci -p no:cacheprovider
                touch $out
              '';
        }
      ];

      ciKernelOutputs = mkKernelOutputs' ciKernels;
      ciRocmKernelOutputs = mkKernelOutputs' ciRocmKernels;
      ciXpuKernelOutputs = mkKernelOutputs' ciXpuKernels;
      ciMetalKernelOutputs = mkKernelOutputs' ciMetalKernels;
      ciCpuKernelOutputs = mkKernelOutputs' ciCpuKernels;
    in
    flake-utils.lib.eachSystem ciKernelsLib.systems (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        mkCiBuild' = mkCiBuild pkgs system;

        ci-build-cuda = mkCiBuild' "ci-test-kernels-cuda" ciKernelOutputs;
        ci-build-rocm = mkCiBuild' "ci-test-kernels-rocm" ciRocmKernelOutputs;
        ci-build-xpu = mkCiBuild' "ci-test-kernels-xpu" ciXpuKernelOutputs;
        ci-build-metal = mkCiBuild' "ci-test-kernels-metal" ciMetalKernelOutputs;
        ci-build-cpu = mkCiBuild' "ci-test-kernels-cpu" ciCpuKernelOutputs;
      in
      {
        packages = {
          inherit
            ci-build-cuda
            ci-build-rocm
            ci-build-xpu
            ci-build-metal
            ci-build-cpu
            ;
          default = ci-build-cuda;
        };
      }
    );
}
