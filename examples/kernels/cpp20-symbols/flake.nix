{
  description = "Flake for invalid-cpp-manylinux-symbols kernel";

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
    };
}
