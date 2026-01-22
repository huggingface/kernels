{
  config,
  lib,
  stdenv,

  cudaSupport ? config.cudaSupport,
  rocmSupport ? config.rocmSupport,
  xpuSupport ? (config.xpuSupport or false),

  callPackage,
  cudaPackages,
  rocmPackages,
}:

{
  xpuPackages,
  version,
}:

let
  system = stdenv.hostPlatform.system;
  flattenVersion = version: lib.replaceStrings [ "." ] [ "" ] (lib.versions.pad 2 version);
  framework =
    if cudaSupport then
      "cu${flattenVersion cudaPackages.cudaMajorMinorVersion}"
    else if rocmSupport then
      "rocm${flattenVersion (lib.versions.majorMinor rocmPackages.rocm.version)}"
    else if xpuSupport then
      "xpu"
    else
      "cpu";
  torchVersions = builtins.fromJSON (builtins.readFile ./torch-versions-hash.json);
  torchBySystem = torchVersions.${version} or (throw "Unsupported torch version: ${version}");
  torchByFramework =
    torchBySystem.${system} or (throw "Unsupported system: ${system} for torch version: ${version}");
  urlHash =
    torchByFramework.${framework}
      or (throw "Unsupported framework: ${framework} for torch version: ${version} on system: ${system}");
in
callPackage ./generic.nix {
  inherit xpuPackages;
  inherit (urlHash) url hash version;
}
