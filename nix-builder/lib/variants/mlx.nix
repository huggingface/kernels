{
  lib,
  buildConfig,
}:

let
  noarch = kernelConfig: "mlx-${buildConfig.backend}-${kernelConfig.toml.mlx-noarch.metal-version}";
in
{
  # There are no arch variants yet.
  arch = noarch;
  inherit noarch;
  kernelVariant = noarch;
}
