{
  nixpkgs,
}:

let
  inherit (nixpkgs) lib;

  overlay = import ../../nix/overlay.nix;

  flattenVersion = version: lib.replaceStrings [ "." ] [ "_" ] (lib.versions.pad 2 version);

  overlayForTorchVersion = torchVersion: sourceBuild: self: super: {
    pythonPackagesExtensions = super.pythonPackagesExtensions ++ [
      (
        python-self: python-super: with python-self; {
          torch =
            if sourceBuild then
              throw "Torch versions with `sourceBuild = true` are not supported anymore"
            else
              python-self."torch-bin_${flattenVersion torchVersion}";
        }
      )
    ];
  };

  # An overlay that overides CUDA to the given version.
  overlayForCudaVersion = cudaVersion: self: super: {
    cudaPackages = super."cudaPackages_${flattenVersion cudaVersion}";
  };

  overlayForRocmVersion = rocmVersion: self: super: {
    rocmPackages = super."rocmPackages_${flattenVersion rocmVersion}";
  };

  overlayForXpuVersion = xpuVersion: self: super: {
    xpuPackages = super."xpuPackages_${flattenVersion xpuVersion}";
  };

  backendConfig = {
    cpu = {
      allowUnfree = true;
    };

    cuda = {
      allowUnfree = true;
      cudaSupport = true;
    };

    metal = {
      allowUnfree = true;
      metalSupport = true;
    };

    rocm = {
      allowUnfree = true;
      rocmSupport = true;
    };

    xpu = {
      allowUnfree = true;
      xpuSupport = true;
    };
  };

  xpuConfig = {
    allowUnfree = true;
    xpuSupport = true;
  };
in

# Construct the nixpkgs package set for the given versions.
buildConfig@{
  backend,
  cpu ? false,
  cudaVersion ? null,
  metal ? false,
  rocmVersion ? null,
  xpuVersion ? null,
  torchVersion,
  system,
  bundleBuild ? false,
  sourceBuild ? false,
}:
let
  backendOverlay =
    if buildConfig.backend == "cpu" then
      [ ]
    else if buildConfig.backend == "cuda" then
      [ (overlayForCudaVersion buildConfig.cudaVersion) ]
    else if buildConfig.backend == "rocm" then
      [ (overlayForRocmVersion buildConfig.rocmVersion) ]
    else if buildConfig.backend == "metal" then
      [ ]
    else if buildConfig.backend == "xpu" then
      [ (overlayForXpuVersion buildConfig.xpuVersion) ]
    else
      throw "No compute framework set in Torch version";
  config =
    backendConfig.${buildConfig.backend} or (throw "No backend config for ${buildConfig.backend}");

  pkgs = import nixpkgs {
    inherit config system;
    overlays = [
      overlay
    ]
    ++ backendOverlay
    ++ [ (overlayForTorchVersion torchVersion sourceBuild) ];
  };

  torch = pkgs.python3.pkgs.torch;

  extension = pkgs.callPackage ./torch-extension { inherit torch; };
in
{
  inherit
    buildConfig
    extension
    pkgs
    torch
    bundleBuild
    ;
}
