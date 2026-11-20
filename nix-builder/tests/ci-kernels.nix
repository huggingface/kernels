# Helpers shared by the flakes that build kernels in CI
# (`examples/kernels/flake.nix` and `nix-builder/tests/kernels/flake.nix`).
#
# Each CI kernel is an attribute set with the following attributes:
#
# - name: name in the output path
# - path: kernel flake path
# - drv (system -> flakeOutputs -> derivation): the derivation for the given
#        system and flake outputs.
# - torchVersions: optional override for the torchVersions argument
# - checkRocmArchs: optional list of ROCm architectures (e.g. "gfx942").
#        When set, the kernel dylib must contain exactly this set of
#        architectures.
# - checkCudaCapabilities: optional list of CUDA capabilities (e.g. "9.0").
#        When set, the kernel dylib must contain exactly this set of
#        capabilities.
# - checkSymbols: optional attrset `{ functions = [ ... ]; layers = [ ... ]; }`.
#        When set, every build variant must contain a symbols.json with
#        exactly these public function and layer names.
# - checkNeeded: optional list of sonames (e.g. "libnccl.so.2"). When set,
#        every kernel dylib must have a NEEDED entry for each of them.
# - checkNotNeeded: optional list of sonames. When set, no kernel dylib
#        may have a NEEDED entry for any of them.
{
  self,
  kernel-builder,
}:

let
  inherit (kernel-builder.inputs.nixpkgs) lib;
in
rec {
  cudaVersion = "cu126";
  cuda13Version = "cu132";
  rocmVersion = "rocm71";
  xpuVersion = "xpu20260";
  torchVersion = "213";
  tvmFfiVersion = "01";

  # Systems that CI kernels are evaluated for.
  systems = [
    "x86_64-linux"
    "aarch64-linux"
    "aarch64-darwin"
  ];

  # Expected public symbols of the relu example kernels.
  reluSymbols = {
    functions = [ "relu" ];
    layers = [ "ReLU" ];
  };

  reluZipSymbols = reluSymbols // {
    functions = [
      "relu"
      "relu_triton"
    ];
  };

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

  # Create a derivation that builds all the given kernels (as returned by
  # `mkKernelOutputs'`) for `system`, running the checks that are requested by
  # the kernels.
  mkCiBuild =
    pkgs: system: name: kernelOutputsList:
    let
      # Check that every *.so in the output of `drv` contains exactly the
      # given set of architectures. `listArchs` is a shell command that
      # prints the architectures of the kernel library `$so`, one per
      # line. On success, the output is a symlink to `drv`.
      checkKernelArchs =
        drv: expectedArchs: listArchs:
        pkgs.runCommand "${drv.name}-check-archs"
          {
            expected = lib.concatLines expectedArchs;
            passAsFile = [ "expected" ];
          }
          ''
            kernelLibs=$(find ${drv}/ -name '*.so')
            if [ -z "$kernelLibs" ]; then
              echo "no *.so files found in ${drv}" >&2
              exit 1
            fi

            sort -u "$expectedPath" > expected
            for so in $kernelLibs; do
              (
                ${listArchs}
              ) | sort -u > actual
              if ! diff -u expected actual; then
                echo "unexpected architectures in $so" >&2
                exit 1
              fi
            done

            ln -s ${drv} $out
          '';

      # Check that every build variant in the output of `drv` contains a
      # symbols.json with exactly the expected public function and layer
      # names. On success, the output is a symlink to `drv`.
      checkSymbols =
        drv: expected:
        pkgs.runCommand "${drv.name}-check-symbols"
          {
            nativeBuildInputs = [ pkgs.jq ];
            expected = builtins.toJSON expected;
            passAsFile = [ "expected" ];
          }
          ''
            variants=$(find ${drv}/ -mindepth 2 -maxdepth 2 -name metadata.json)
            if [ -z "$variants" ]; then
              echo "no build variants found in ${drv}" >&2
              exit 1
            fi

            jq -S . "$expectedPath" > expected
            for metadata in $variants; do
              symbols="$(dirname "$metadata")/symbols.json"
              if [ ! -f "$symbols" ]; then
                echo "missing $symbols" >&2
                exit 1
              fi
              jq -S '{functions: [.functions[].name], layers: [.layers[].name]}' "$symbols" > actual
              if ! diff -u expected actual; then
                echo "unexpected symbols in $symbols" >&2
                exit 1
              fi
            done

            ln -s ${drv} $out
          '';

      # Check that every *.so in the output of `drv` has a NEEDED entry for
      # each soname in `needed` and none for the sonames in `notNeeded`. On
      # success, the output is a symlink to `drv`.
      checkNeededLibs =
        drv:
        {
          needed ? [ ],
          notNeeded ? [ ],
        }:
        pkgs.runCommand "${drv.name}-check-needed"
          {
            nativeBuildInputs = [ pkgs.binutils ];
            needed = lib.concatLines needed;
            notNeeded = lib.concatLines notNeeded;
            passAsFile = [
              "needed"
              "notNeeded"
            ];
          }
          ''
            kernelLibs=$(find ${drv}/ -name '*.so')
            if [ -z "$kernelLibs" ]; then
              echo "no *.so files found in ${drv}" >&2
              exit 1
            fi

            for so in $kernelLibs; do
              readelf -d "$so" | sed -n 's/.*(NEEDED).*Shared library: \[\(.*\)\]/\1/p' | sort -u > actual

              while read -r soname; do
                [ -z "$soname" ] && continue
                if ! grep -qxF "$soname" actual; then
                  echo "$so does not have a NEEDED entry for $soname" >&2
                  exit 1
                fi
              done < "$neededPath"

              while read -r soname; do
                [ -z "$soname" ] && continue
                if grep -qxF "$soname" actual; then
                  echo "$so unexpectedly has a NEEDED entry for $soname" >&2
                  exit 1
                fi
              done < "$notNeededPath"
            done

            ln -s ${drv} $out
          '';

      checkRocmArchs =
        drv: expectedArchs:
        checkKernelArchs drv expectedArchs
          # Use the ROCm LLVM that the kernel was built with. Stock
          # llvm-readobj does not support `--offloading`.
          ''
            ${drv.torch.rocmPackages.rocm-llvm}/llvm/bin/llvm-readobj --offloading "$so" \
                | grep -oE 'gfx[0-9]+[a-z]?'
          '';

      checkCudaCapabilities =
        drv: expectedCapabilities:
        let
          # Convert capabilities like "9.0a" to sm names like "sm_90a".
          smArchs = map (cap: "sm_" + lib.replaceStrings [ "." ] [ "" ] cap) expectedCapabilities;
        in
        checkKernelArchs drv smArchs ''
          ${drv.torch.cudaPackages.cuda_cuobjdump}/bin/cuobjdump "$so" \
              | grep -oE 'sm_[0-9]+[af]?'
        '';

      resolveKernels = map (kernel: {
        inherit (kernel) name;
        drv =
          let
            baseDrv = kernel.drv system kernel.outputs;
            archsDrv =
              if kernel ? checkRocmArchs then
                checkRocmArchs baseDrv kernel.checkRocmArchs
              else if kernel ? checkCudaCapabilities then
                checkCudaCapabilities baseDrv kernel.checkCudaCapabilities
              else
                baseDrv;
            neededDrv =
              if kernel ? checkNeeded || kernel ? checkNotNeeded then
                checkNeededLibs archsDrv {
                  needed = kernel.checkNeeded or [ ];
                  notNeeded = kernel.checkNotNeeded or [ ];
                }
              else
                archsDrv;
          in
          if kernel.assertFail or false then
            pkgs.testers.testBuildFailure' {
              drv = baseDrv;
              expectedBuilderLogEntries = kernel.assertFailLogs or [ ];
            }
          else if kernel ? checkSymbols then
            checkSymbols neededDrv kernel.checkSymbols
          else
            neededDrv;
      }) kernelOutputsList;
    in
    pkgs.linkFarm name (
      map (kernel: {
        inherit (kernel) name;
        path = kernel.drv;
      }) resolveKernels
    );
}
