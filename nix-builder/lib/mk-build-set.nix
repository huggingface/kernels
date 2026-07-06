{
  nixpkgs,
  rust-overlay,
}:

let
  inherit (nixpkgs) lib;

  overlay = import ../overlay.nix;

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
  overlayForCudaVersion = cudaVersion: ptxasVersion: self: super: {
    cudaPackages =
      let
        cudaPackages' = super."cudaPackages_${flattenVersion cudaVersion}";
        ptxasPackages = super."cudaPackages_${flattenVersion ptxasVersion}";
      in
      if ptxasVersion == cudaVersion then
        cudaPackages'
      else
        cudaPackages'.overrideScope (
          self: super: {
            cuda_nvcc = super.cuda_nvcc.overrideAttrs (prevAttrs: {
              # Do this before the original postInstall, so that subsequent
              # postInstall steps are applied.
              postInstall = ''
                cp ${ptxasPackages.cuda_nvcc.src}/bin/ptxas $out/bin/ptxas
                cp ${ptxasPackages.cuda_nvcc.src}/bin/nvlink $out/bin/nvlink
                cp ${ptxasPackages.cuda_nvcc.src}/nvvm/bin/* $out/nvvm/bin/
              ''
              + prevAttrs.postInstall or "";
            });
          }
        );

  };

  overlayForRocmVersion = rocmVersion: self: super: {
    rocmPackages = super."rocmPackages_${flattenVersion rocmVersion}";
  };

  overlayForXpuVersion = xpuVersion: self: super: {
    xpuPackages = super."xpuPackages_${lib.replaceStrings [ "." ] [ "_" ] xpuVersion}";
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
  ptxasVersion ? cudaVersion,
  metal ? false,
  rocmVersion ? null,
  xpuVersion ? null,
  torchVersion,
  system,
  bundleBuild ? false,
  sourceBuild ? false,
  tvmFfiVersion ? null,
}:
let
  backendOverlay =
    if buildConfig.backend == "cpu" then
      [ ]
    else if buildConfig.backend == "cuda" then
      [ (overlayForCudaVersion cudaVersion ptxasVersion) ]
    else if buildConfig.backend == "rocm" then
      [ (overlayForRocmVersion rocmVersion) ]
    else if buildConfig.backend == "metal" then
      [ ]
    else if buildConfig.backend == "xpu" then
      [ (overlayForXpuVersion xpuVersion) ]
    else
      throw "No compute framework set in Torch version";
  config =
    backendConfig.${buildConfig.backend} or (throw "No backend config for ${buildConfig.backend}");

  pkgs = import nixpkgs {
    inherit config system;
    overlays = [
      overlay
      rust-overlay.overlays.default
    ]
    ++ backendOverlay
    ++ [ (overlayForTorchVersion torchVersion sourceBuild) ];
  };

  torch = pkgs.python3.pkgs.torch;

  extension = pkgs.callPackage ./extension { inherit torch; };

  variants = import ./variants {
    inherit lib buildConfig;
  };
in
{
  inherit
    buildConfig
    extension
    pkgs
    torch
    bundleBuild
    variants
    ;
}
