{
  description = "All example kernels";

  inputs = {
    kernel-builder.url = "path:../..";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    let
      inherit (kernel-builder.inputs) flake-utils nixpkgs;
      inherit (kernel-builder.inputs.nixpkgs) lib;

      cudaVersion = "cu126";
      rocmVersion = "rocm71";
      xpuVersion = "xpu20253";
      torchVersion = "211";
      tvmFfiVersion = "01";

      # All example kernels to build in CI.
      #
      # - name: name in the output path
      # - path: kernel flake path
      # - drv (system -> flakeOutputs -> derivation): the derivation for the given
      #        system and flake outputs.
      # - torchVersions: optional override for the torchVersions argument
      ciKernels = [
        {
          name = "cpp20-symbols-kernel";
          path = ./cpp20-symbols;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-cpu-${sys}"};
        }
        {
          name = "relu-kernel";
          path = ./relu;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-${cudaVersion}-${sys}"};
        }
        {
          name = "relu-torch-stable-abi-kernel";
          path = ./relu-torch-stable-abi;
          drv =
            sys: out:
            out.packages.${sys}.redistributable.${"torch-stable-abi${torchVersion}-${cudaVersion}-${sys}"};
        }
        {
          name = "relu-tvm-ffi-kernel";
          path = ./relu-tvm-ffi;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"tvm-ffi${tvmFfiVersion}-${cudaVersion}-${sys}"};
        }
        {
          name = "relu-tvm-ffi-compiler-flags-kernel";
          path = ./relu-tvm-ffi-compiler-flags;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"tvm-ffi${tvmFfiVersion}-${cudaVersion}-${sys}"};
        }
        {
          name = "extra-data";
          path = ./extra-data;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-${cudaVersion}-${sys}"};
        }
        {
          name = "relu-kernel-cpu";
          path = ./relu;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-cpu-${sys}"};
        }
        {
          name = "cutlass-gemm-kernel";
          path = ./cutlass-gemm;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-${cudaVersion}-${sys}"};
        }
        {
          name = "cutlass-gemm-tvm-ffi-kernel";
          path = ./cutlass-gemm-tvm-ffi;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"tvm-ffi${tvmFfiVersion}-${cudaVersion}-${sys}"};
        }
        {
          name = "relu-backprop-compile-kernel";
          path = ./relu-backprop-compile;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-${cudaVersion}-${sys}"};
        }
        {
          name = "silu-and-mul-kernel";
          path = ./silu-and-mul;
          drv = sys: out: out.packages.${sys}.redistributable.torch-cuda;
        }
        {
          # Tests that we can build with the extra torchVersions argument.
          name = "relu-specific-torch";
          path = ./relu-specific-torch;
          drv = sys: out: out.packages.${sys}.default;
          torchVersions = _defaultVersions: [
            {
              torchVersion = "2.11";
              cudaVersion = "12.8";
              systems = [
                "x86_64-linux"
                "aarch64-linux"
              ];
              bundleBuild = true;
            }
          ];
        }
        {
          name = "relu-compiler-flags";
          path = ./relu-compiler-flags;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-${cudaVersion}-${sys}"};
        }
        {
          name = "relu-invalid-capability";
          path = ./relu-invalid-capability;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-${cudaVersion}-${sys}"};
          assertFail = true;
          assertFailLogs = [ "empty set of capabilities" ];
        }
        {
          # Check that we can build an arch dev shell.
          name = "relu-dev-shell";
          path = ./relu;
          drv = sys: out: out.devShells.${sys}.default;
        }
        {
          # Check that we can build an arch test shell (e.g. gcc is
          # compatible with CUDA requirements).
          name = "relu-test-shell";
          path = ./relu;
          drv = sys: out: out.devShells.${sys}.test;
        }
        {
          # Check that we can build a noarch dev shell.
          name = "silu-and-mul-dev-shell";
          path = ./silu-and-mul;
          drv = sys: out: out.devShells.${sys}.default;
        }
        {
          # Check that we can build an noarch test shell.
          name = "silu-and-mul-test-shell";
          path = ./silu-and-mul;
          drv = sys: out: out.devShells.${sys}.test;
        }
        {
          name = "relu-triton-kernel";
          path = ./relu-triton;
          drv = sys: out: out.packages.${sys}.redistributable.torch-cuda;
        }
      ];

      # ROCm kernels to build in CI.
      ciRocmKernels = [
        {
          name = "relu-triton-kernel";
          path = ./relu-triton;
          drv = sys: out: out.packages.${sys}.redistributable.torch-rocm;
        }
        {
          name = "relu-invalid-capability";
          path = ./relu-invalid-capability;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-${rocmVersion}-${sys}"};
          assertFail = true;
          assertFailLogs = [ "empty set of architectures" ];
        }
        {
          name = "relu-kernel";
          path = ./relu;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-${rocmVersion}-${sys}"};
        }
        {
          name = "relu-compiler-flags";
          path = ./relu-compiler-flags;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-${rocmVersion}-${sys}"};
        }
      ];

      mkKernelOutputs =
        {
          path,
          torchVersions ? null,
        }:
        kernel-builder.lib.genKernelFlakeOutputs (
          {
            inherit self path;
          }
          // lib.optionalAttrs (torchVersions != null) { inherit torchVersions; }
        );

      mkKernelOutputs' =
        kernels:
        map (
          kernel:
          kernel
          // {
            outputs = mkKernelOutputs {
              inherit (kernel) path;
              torchVersions = kernel.torchVersions or null;
            };
          }
        ) kernels;

      ciKernelOutputs = mkKernelOutputs' ciKernels;
      ciRocmKernelOutputs = mkKernelOutputs' ciRocmKernels;

      # XPU kernels to build in CI.
      ciXpuKernels = [
        {
          name = "relu-triton-kernel";
          path = ./relu-triton;
          drv = sys: out: out.packages.${sys}.redistributable.torch-xpu;
        }
        {
          name = "relu-kernel";
          path = ./relu;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-${xpuVersion}-${sys}"};
        }
        {
          name = "relu-tvm-ffi-kernel";
          path = ./relu-tvm-ffi;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"tvm-ffi${tvmFfiVersion}-${xpuVersion}-${sys}"};
        }
        {
          name = "relu-tvm-ffi-compiler-flags-kernel";
          path = ./relu-tvm-ffi-compiler-flags;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"tvm-ffi${tvmFfiVersion}-${xpuVersion}-${sys}"};
        }
        {
          name = "relu-compiler-flags";
          path = ./relu-compiler-flags;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-${xpuVersion}-${sys}"};
        }
        {
          name = "cutlass-gemm-kernel";
          path = ./cutlass-gemm;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-${xpuVersion}-${sys}"};
        }
      ];

      # Metal kernels to build in CI.
      ciMetalKernels = [
        {
          name = "relu-kernel";
          path = ./relu;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-metal-${sys}"};
        }
        {
          name = "relu-metal-cpp-kernel";
          path = ./relu-metal-cpp;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-metal-${sys}"};
        }
      ];

      ciXpuKernelOutputs = mkKernelOutputs' ciXpuKernels;
      ciMetalKernelOutputs = mkKernelOutputs' ciMetalKernels;
    in
    flake-utils.lib.eachSystem
      [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ]
      (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};

          resolveKernels =
            kernelOutputsList:
            map (kernel: {
              inherit (kernel) name;
              drv =
                let
                  baseDrv = kernel.drv system kernel.outputs;
                in
                if kernel.assertFail or false then
                  pkgs.testers.testBuildFailure' {
                    drv = baseDrv;
                    expectedBuilderLogEntries = kernel.assertFailLogs or [ ];
                  }
                else
                  baseDrv;
            }) kernelOutputsList;

          mkCiBuild =
            name: kernelOutputsList:
            pkgs.linkFarm name (
              map (kernel: {
                inherit (kernel) name;
                path = kernel.drv;
              }) (resolveKernels kernelOutputsList)
            );

          ci-build-cuda = mkCiBuild "ci-kernels-cuda" ciKernelOutputs;
          ci-build-rocm = mkCiBuild "ci-kernels-rocm" ciRocmKernelOutputs;
          ci-build-xpu = mkCiBuild "ci-kernels-xpu" ciXpuKernelOutputs;
          ci-build-metal = mkCiBuild "ci-kernels-metal" ciMetalKernelOutputs;
        in
        {
          packages = {
            inherit
              ci-build-cuda
              ci-build-rocm
              ci-build-xpu
              ci-build-metal
              ;
            default = ci-build-cuda;
          };
        }
      );
}
