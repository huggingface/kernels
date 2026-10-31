{
  description = "Flake for ReLU kernel";

  inputs = {
    kernel-builder.url = "path:../../..";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
      torchVersions = defaultVersions: [
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
    };
}
