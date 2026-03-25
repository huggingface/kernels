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
      torchVersion = "29";
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
          name = "relu-kernel";
          path = ./relu;
          drv =
            sys: out: out.packages.${sys}.redistributable.${"torch${torchVersion}-cxx11-${cudaVersion}-${sys}"};
        }
        {
          name = "relu-tvm-ffi-kernel";
          path = ./relu-tvm-ffi;
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
              torchVersion = "2.9";
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
          # Check that we can build a test shell (e.g. gcc is compatible with
          # CUDA requirements).
          name = "relu-test-shell";
          path = ./relu;
          drv = sys: out: out.devShells.${sys}.test;
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

      ciKernelOutputs = map (
        kernel:
        kernel
        // {
          outputs = mkKernelOutputs {
            inherit (kernel) path;
            torchVersions = kernel.torchVersions or null;
          };
        }
      ) ciKernels;
    in
    flake-utils.lib.eachSystem
      [
        "x86_64-linux"
        "aarch64-linux"
      ]
      (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};

          resolvedKernels = map (kernel: {
            inherit (kernel) name;
            drv = kernel.drv system kernel.outputs;
          }) ciKernelOutputs;

          ci-build = pkgs.linkFarm "ci-kernels" (
            map (kernel: {
              inherit (kernel) name;
              path = kernel.drv;
            }) resolvedKernels
          );
        in
        {
          packages = {
            inherit ci-build;
            default = ci-build;
          };
        }
      );
}
