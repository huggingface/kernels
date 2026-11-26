{
  description = "Arch SDK without a backend (test)";

  inputs = {
    kernel-builder.url = "path:../../../..";
  };

  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genSdkFlakeOutputs {
      inherit self;

      name = "sdk-arch";
      repoId = "kernels-test/sdk-arch";

      python = {
        version = "3.9";
        abi = "abi3";
      };
      useBackend = false;

      mkVariant =
        { buildSet, variant, ... }:
        buildSet.pkgs.callPackage ./sdk.nix {
          inherit variant;
          backend = "cpu";
        };

      mkTests = _: ./tests;
    };
}
