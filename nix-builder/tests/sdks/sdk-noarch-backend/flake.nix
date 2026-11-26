{
  description = "Noarch SDK with a variant per backend (test)";

  inputs = {
    kernel-builder.url = "path:../../../..";
  };

  outputs =
    { self, kernel-builder }:
    kernel-builder.lib.genSdkFlakeOutputs {
      inherit self;

      name = "sdk-noarch-backend";
      repoId = "kernels-test/sdk-noarch-backend";

      python = null;

      mkVariant =
        {
          buildSet,
          variant,
          sdkId,
        }:
        buildSet.pkgs.callPackage ./sdk.nix {
          inherit variant sdkId;
          inherit (buildSet.buildConfig) backend;
        };

      mkTests = _: ./tests;
    };
}
