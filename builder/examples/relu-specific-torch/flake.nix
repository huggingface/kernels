{
  description = "Flake for ReLU kernel";

  inputs = {
    kernel-builder.url = "path:../..";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genFlakeOutputs {
      inherit self;
      path = ./.;
      torchVersions = defaultVersions: [
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
    };
}
