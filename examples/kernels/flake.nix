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

      ciKernelsLib = import "${kernel-builder}/nix-builder/tests/ci-kernels.nix" {
        inherit self kernel-builder;
      };

      inherit (ciKernelsLib)
        cudaVersion
        cuda13Version
        rocmVersion
        xpuVersion
        torchVersion
        tvmFfiVersion
        reluSymbols
        mkKernelOutputs'
        mkCiBuild
        ;

      # All example kernels to build in CI.
      ciKernels = [
        # This test should check the capabilities of the oldest supported CUDA.
        {
          name = "relu-kernel";
          path = ./relu;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${cudaVersion}-${sys}"};
          checkSymbols = reluSymbols;
          # Kernels that do not use NCCL must not link against it.
          checkNotNeeded = [ "libnccl.so.2" ];
          checkCudaCapabilities = [
            "7.0"
            "7.2"
            "7.5"
            "8.0"
            "8.6"
            "8.7"
            "8.9"
            "9.0"
          ];
        }
        # This test should check the capabilities of the latest supported CUDA.
        {
          name = "relu-kernel-cu13";
          path = ./relu;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${cuda13Version}-${sys}"};
          checkCudaCapabilities = [
            "7.5"
            "8.0"
            "8.6"
            "8.7"
            "8.9"
            "9.0"
            "10.0"
            "10.3"
            "11.0"
            "12.0"
            "12.1"
          ];
        }
        {
          name = "relu-torch-stable-abi-kernel";
          path = ./relu-torch-stable-abi;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch-stable-abi211-${cudaVersion}-${sys}"};
        }
        {
          # Exercise the shim with Torch 2.11 headers, which lack index_select.
          name = "index-select-torch-stable-abi-kernel";
          path = ./index-select-torch-stable-abi;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch-stable-abi210-cu128-${sys}"};
        }
        # Check that we can also build the non-stable ABI version preceding the stable ABI version.
        {
          name = "relu-torch-stable-abi-kernel";
          path = ./relu-torch-stable-abi;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch210-${cudaVersion}-${sys}"};
        }
        {
          name = "relu-tvm-ffi-kernel";
          path = ./relu-tvm-ffi;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"tvm-ffi${tvmFfiVersion}-${cudaVersion}-${sys}"};
          checkSymbols = {
            functions = [
              "relu"
              "relu_jax"
            ];
            layers = [ ];
          };
        }
        {
          name = "relu-rust-cuda-oxide-kernel";
          path = ./relu-rust-cuda-oxide;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"tvm-ffi${tvmFfiVersion}-${cuda13Version}-${sys}"};
        }
        {
          name = "relu-rust-cutile-kernel";
          path = ./relu-rust-cutile;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"tvm-ffi${tvmFfiVersion}-${cuda13Version}-${sys}"};
          checkSymbols = {
            functions = [ "relu" ];
            layers = [ ];
          };
        }
        {
          name = "relu-kernel-cpu";
          path = ./relu;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cpu-${sys}"};
        }
        {
          name = "cutlass-gemm-kernel";
          path = ./cutlass-gemm;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${cudaVersion}-${sys}"};
        }
        {
          name = "relu-backprop-compile-kernel";
          path = ./relu-backprop-compile;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${cudaVersion}-${sys}"};
        }
        {
          name = "silu-and-mul-kernel";
          path = ./silu-and-mul;
          drv = sys: out: out.packages.${sys}.redistributable.torch-cuda;
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
          name = "relu-kernel";
          path = ./relu;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${rocmVersion}-${sys}"};
          checkRocmArchs = [
            "gfx906"
            "gfx908"
            "gfx90a"
            "gfx942"
            "gfx950"
            "gfx1030"
            "gfx1100"
            "gfx1101"
            "gfx1102"
            "gfx1103"
            "gfx1150"
            "gfx1151"
            "gfx1200"
            "gfx1201"
          ];
        }
        {
          name = "relu-tvm-ffi-kernel";
          path = ./relu-tvm-ffi;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"tvm-ffi${tvmFfiVersion}-${rocmVersion}-${sys}"};
        }
        {
          name = "relu-torch-stable-abi-kernel";
          path = ./relu-torch-stable-abi;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch-stable-abi211-${rocmVersion}-${sys}"};
        }
        # Check that we can also build the non-stable ABI version preceding the stable ABI version.
        {
          name = "relu-torch-stable-abi-kernel";
          path = ./relu-torch-stable-abi;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch210-${rocmVersion}-${sys}"};
        }
      ];

      # XPU kernels to build in CI.
      ciXpuKernels = [
        {
          name = "relu-triton-kernel";
          path = ./relu-triton;
          drv = sys: out: out.packages.${sys}.redistributable.torch-xpu;
          checkSymbols = reluSymbols;
        }
        {
          name = "relu-kernel";
          path = ./relu;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${xpuVersion}-${sys}"};
          checkSymbols = reluSymbols;
        }
        {
          name = "relu-tvm-ffi-kernel";
          path = ./relu-tvm-ffi;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"tvm-ffi${tvmFfiVersion}-${xpuVersion}-${sys}"};
          checkSymbols = {
            functions = [
              "relu"
              "relu_jax"
            ];
            layers = [ ];
          };
        }
        {
          name = "cutlass-gemm-kernel";
          path = ./cutlass-gemm;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-${xpuVersion}-${sys}"};
        }
        {
          name = "relu-torch-stable-abi-kernel";
          path = ./relu-torch-stable-abi;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch-stable-abi211-${xpuVersion}-${sys}"};
        }
        # Check that we can also build the non-stable ABI version preceding the stable ABI version.
        {
          name = "relu-torch-stable-abi-kernel";
          path = ./relu-torch-stable-abi;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch210-xpu20253-${sys}"};
        }
      ];

      # TPU kernels to build in CI.
      ciTpuKernels = [
        {
          name = "relu-tpu-kernel";
          path = ./relu-tpu;
          drv = sys: out: out.packages.${sys}.redistributable.torch-tpu;
        }
      ];

      # Metal kernels to build in CI.
      ciMetalKernels = [
        {
          name = "relu-kernel";
          path = ./relu;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-metal-${sys}"};
          checkSymbols = reluSymbols;
        }
        {
          name = "relu-metal-cpp-kernel";
          path = ./relu-metal-cpp;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-metal-${sys}"};
        }
        {
          name = "relu-mlx-kernel";
          path = ./relu-mlx;
          drv = sys: out: out.packages.${sys}.redistributable."mlx-metal";
        }
      ];

      # CPU kernels to build in CI.
      ciCpuKernels = [
        {
          name = "relu-rust-kernel";
          path = ./relu-rust;
          drv =
            sys: out:
            let
              variant = "tvm-ffi${tvmFfiVersion}-cpu-${sys}";
              extension = out.packages.${sys}.redistributable.${variant};
              ciTest = out.packages.${sys}.ciTests.${variant};
              kernelPkgs = out.packages.${sys}.pkgs.${variant};
            in
            kernelPkgs.runCommand "relu-rust-kernel-test"
              {
                nativeBuildInputs = [ ciTest ];
              }
              ''
                ${ciTest}/bin/ci-test
                ln -s ${extension} $out
              '';
        }
        {
          name = "relu-torch-stable-abi-kernel";
          path = ./relu-torch-stable-abi;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch-stable-abi211-cpu-${sys}"};
        }
        # Check that we can also build the non-stable ABI version preceding the stable ABI version.
        {
          name = "relu-torch-stable-abi-kernel";
          path = ./relu-torch-stable-abi;
          drv = sys: out: out.packages.${sys}.redistributable.${"torch210-cpu-${sys}"};
        }
      ];

      ciKernelOutputs = mkKernelOutputs' ciKernels;
      ciRocmKernelOutputs = mkKernelOutputs' ciRocmKernels;
      ciXpuKernelOutputs = mkKernelOutputs' ciXpuKernels;
      ciTpuKernelOutputs = mkKernelOutputs' ciTpuKernels;
      ciMetalKernelOutputs = mkKernelOutputs' ciMetalKernels;
      ciCpuKernelOutputs = mkKernelOutputs' ciCpuKernels;
    in
    flake-utils.lib.eachSystem ciKernelsLib.systems (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        mkCiBuild' = mkCiBuild pkgs system;

        ci-build-cuda = mkCiBuild' "ci-kernels-cuda" ciKernelOutputs;
        ci-build-rocm = mkCiBuild' "ci-kernels-rocm" ciRocmKernelOutputs;
        ci-build-xpu = mkCiBuild' "ci-kernels-xpu" ciXpuKernelOutputs;
        ci-build-tpu = mkCiBuild' "ci-kernels-tpu" ciTpuKernelOutputs;
        ci-build-metal = mkCiBuild' "ci-kernels-metal" ciMetalKernelOutputs;
        ci-build-cpu = mkCiBuild' "ci-kernels-cpu" ciCpuKernelOutputs;
      in
      {
        packages = {
          inherit
            ci-build-cuda
            ci-build-rocm
            ci-build-xpu
            ci-build-tpu
            ci-build-metal
            ci-build-cpu
            ;
          default = ci-build-cuda;
        };
      }
    );
}
