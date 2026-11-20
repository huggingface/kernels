{
  lib,
  buildConfig,
}:

let
  noarch = _kernelConfig: "mlx-${buildConfig.backend}";
in
{
  # There are no arch variants yet.
  arch = noarch;
  inherit noarch;
  kernelVariant = noarch;
}
