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
  torchString =
    stableAbi:
    if stableAbi == null then
      "torch${flattenVersion (lib.versions.majorMinor buildConfig.torchVersion)}"
    else
      "torch-stable-abi${flattenVersion (lib.versions.majorMinor stableAbi)}";
  archString =
    if buildConfig.system == "aarch64-darwin" then
      "${torchString null}-${computeString}-${buildConfig.system}"
    else
      "${torchString null}-cxx11-${computeString}-${buildConfig.system}";
in
{
  arch = archString;
  noarch = "torch-${buildConfig.backend}";

  kernelVariant =
    kernelConfig:
    let
      archVariant = kernelConfig.kernelBackends.${buildConfig.backend};
    in
    if archVariant && kernelConfig.isTorchStableAbi then
      "torch-stable-abi${flattenVersion (lib.versions.majorMinor kernelConfig.torchStableAbiVersion)}-${computeString}-${buildConfig.system}"
    else if archVariant then
      archString
    else
      "torch-${buildConfig.backend}";

}
