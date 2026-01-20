{
  config,
  cudaSupport ? config.cudaSupport,
  metalSupport ? config.metalSupport or false,
  rocmSupport ? config.rocmSupport,
  xpuSupport ? config.xpuSupport or false,

  cudaPackages,
  rocmPackages,
  xpuPackages,

  lib,
  stdenv,

  torchVersion,
}:

let
  flattenVersion =
    version: lib.replaceStrings [ "." ] [ "" ] (lib.versions.majorMinor (lib.versions.pad 2 version));
  backend =
    if cudaSupport then
      "cuda"
    else if metalSupport then
      "metal"
    else if rocmSupport then
      "rocm"
    else if xpuSupport then
      "xpu"
    else
      "cpu";
  computeString =
    if cudaSupport then
      "cu${flattenVersion cudaPackages.cudaMajorMinorVersion}"
    else if metalSupport then
      "metal"
    else if rocmSupport then
      "rocm${flattenVersion (lib.versions.majorMinor rocmPackages.rocm.version)}"
    else if xpuSupport then
      "xpu${flattenVersion (lib.versions.majorMinor xpuPackages.oneapi-torch-dev.version)}"
    else
      "cpu";
in
{
  variant =
    if stdenv.hostPlatform.system == "aarch64-darwin" then
      "torch${flattenVersion (lib.versions.majorMinor torchVersion)}-${computeString}-${stdenv.hostPlatform.system}"
    else
      "torch${flattenVersion (lib.versions.majorMinor torchVersion)}-cxx11-${computeString}-${stdenv.hostPlatform.system}";

  noarchVariant = "torch-${backend}";
}
