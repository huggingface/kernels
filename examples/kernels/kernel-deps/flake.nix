{
  description = "Flake for kernels tests";

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

      # TODO: enable once we expose dependencies in Nix.
      doGetKernelCheck = false;
    };
}
