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
  torchString = "torch${flattenVersion (lib.versions.majorMinor buildConfig.torchVersion)}";
  archString =
    if buildConfig.system == "aarch64-darwin" then
      "${torchString}-${computeString}-${buildConfig.system}"
    else
      "${torchString}-cxx11-${computeString}-${buildConfig.system}";
in
{
  arch = archString;
  noarch = "torch-${buildConfig.backend}";

  kernelVariant =
    kernelConfig:
    let
      archVariant = kernelConfig.kernelBackends.${buildConfig.backend};
      stableAbiVersion = kernelConfig.torchStableAbiVersionForBackend buildConfig.backend;
    in
    if archVariant && stableAbiVersion != null then
      "torch-stable-abi${flattenVersion (lib.versions.majorMinor stableAbiVersion)}-${computeString}-${buildConfig.system}"
    else if archVariant then
      archString
    else
      "torch-${buildConfig.backend}";

}
