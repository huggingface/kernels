{
  lib,
  buildConfig,
}:

let
  noarch = "mlx-${buildConfig.backend}";
in
{
  # There are no arch variants yet.
  arch = noarch;
  inherit noarch;
  kernelVariant = _: noarch;
}
