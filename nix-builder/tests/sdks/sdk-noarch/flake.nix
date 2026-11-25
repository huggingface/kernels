{
  description = "Noarch SDK without a backend (test)";

  inputs = {
    kernel-builder.url = "path:../../../..";
  };

  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genSdkFlakeOutputs {
      inherit self;

      name = "sdk-noarch";
      repoId = "kernels-test/sdk-noarch";

      python = null;
      useBackend = false;

      mkVariant =
        { buildSet, variant, ... }:
        buildSet.pkgs.callPackage ./sdk.nix {
          inherit variant;
          # The metadata requires a backend.
          backend = "cpu";
        };

      mkTests = _: ./tests;
    };
}
