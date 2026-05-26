{
  lib,
  buildConfig,
}:

rec {
  torch = import ./torch.nix { inherit lib buildConfig; };
  tvm-ffi = import ./tvm-ffi.nix { inherit lib buildConfig; };

  kernelVariant =
    kernelConfig:
    if kernelConfig.isTvmFfi then
      tvm-ffi.kernelVariant kernelConfig
    else
      torch.kernelVariant kernelConfig;

  kernelArchVariant = kernelConfig: if kernelConfig.isTvmFfi then tvm-ffi.arch else torch.arch;
}
