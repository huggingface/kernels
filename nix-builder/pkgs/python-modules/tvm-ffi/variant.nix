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

  tvmFfiVersion,
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
  variant = "tvm-ffi${flattenVersion (lib.versions.majorMinor tvmFfiVersion)}-${computeString}-${stdenv.hostPlatform.system}";
}
