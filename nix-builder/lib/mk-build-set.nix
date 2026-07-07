{
  nixpkgs,
  rust-overlay,

  # Git provenance (`{ sha, dirty }` or `null`) of the `kernel-builder` flake,
  # so it can be burned into the binary that the kernels it builds record.
  builderProvenance ? null,
}:

let
  inherit (nixpkgs) lib;

  overlay = import ../overlay.nix { inherit builderProvenance; };

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

    tpu = {
      # torch_tpu is Apache-2.0, but libtpu's wheel METADATA declares
      # its license as "Google Cloud Platform Terms of Service"
      # (unfree), so the tpu buildSet needs allowUnfree just like the
      # cuda/rocm/xpu sets. See pkgs/python-modules/libtpu/default.nix.
      allowUnfree = true;
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
    else if buildConfig.backend == "tpu" then
      [ ]
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

  extension = pkgs.callPackage ./extension {
    inherit torch;
    python3 = if buildConfig.backend == "tpu" then pkgs.python312 else pkgs.python3;
  };

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
