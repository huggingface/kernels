{
  lib,
  buildConfig,
}:

rec {
  mlx = import ./mlx.nix { inherit lib buildConfig; };
  torch = import ./torch.nix { inherit lib buildConfig; };
  tvm-ffi = import ./tvm-ffi.nix { inherit lib buildConfig; };

  kernelVariant =
    kernelConfig:
    if kernelConfig.isMlxNoarch then
      mlx.kernelVariant kernelConfig
    else if kernelConfig.isTvmFfi then
      tvm-ffi.kernelVariant kernelConfig
    else
      torch.kernelVariant kernelConfig;

  kernelArchVariant =
    kernelConfig:
    if kernelConfig.isMlxNoarch then
      mlx.arch
    else if kernelConfig.isTvmFfi then
      tvm-ffi.arch
    else
      torch.arch;
}
