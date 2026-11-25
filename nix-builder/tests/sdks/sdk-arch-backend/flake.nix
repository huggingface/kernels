{
  description = "Arch SDK with a backend (test)";

  inputs = {
    kernel-builder.url = "path:../../../..";
  };

  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genSdkFlakeOutputs {
      inherit self;

      name = "sdk-arch-backend";
      repoId = "kernels-test/sdk-arch-backend";

      python = {
        version = "3.9";
        abi = "abi3";
      };

      # Only build for CUDA (one version, to keep the test cheap).
      torchVersions =
        versions: builtins.filter (version: (version.cudaVersion or null) == "12.6") versions;

      mkVariant =
        { buildSet, variant, ... }:
        buildSet.pkgs.callPackage ./sdk.nix {
          inherit variant;
          inherit (buildSet.buildConfig) backend;
        };

      tests = ./tests;
    };
}
