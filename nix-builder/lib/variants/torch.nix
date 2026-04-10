{
  lib,
  buildConfig,
}:

let
  flattenVersion =
    version: lib.replaceStrings [ "." ] [ "" ] (lib.versions.majorMinor (lib.versions.pad 2 version));
  computeString =
    if buildConfig.backend == "cuda" then
      "cu${flattenVersion buildConfig.cudaVersion}"
    else if buildConfig.backend == "metal" then
      "metal"
    else if buildConfig.backend == "rocm" then
      "rocm${flattenVersion (lib.versions.majorMinor buildConfig.rocmVersion)}"
    else if buildConfig.backend == "xpu" then
      "xpu${flattenVersion (lib.versions.majorMinor buildConfig.xpuVersion)}"
    else
      "cpu";
in
{
  arch =
    if buildConfig.system == "aarch64-darwin" then
      "torch${flattenVersion (lib.versions.majorMinor buildConfig.torchVersion)}-${computeString}-${buildConfig.system}"
    else
      "torch${flattenVersion (lib.versions.majorMinor buildConfig.torchVersion)}-cxx11-${computeString}-${buildConfig.system}";

  noarch = "torch-${buildConfig.backend}";
}
