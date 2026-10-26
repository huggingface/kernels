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

  # The nixpkgs instance is shared between build sets with different Torch
  # versions, so it must not provide a Torch. Torch is available through
  # the Python 3 package set accessible through `buildSet.python3`.
  noTorchOverlay = self: super: {
    pythonPackagesExtensions = super.pythonPackagesExtensions ++ [
      (python-self: python-super: {
        torch = throw "`python3.pkgs.torch` is not available in the shared package set, use `buildSet.torch` or `buildSet.python3`";
      })
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
      tpuSupport = true;
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

{
  # Key that identifies the nixpkgs instance for a build config. Build
  # configs with the same key can share a nixpkgs instance.
  pkgsKey =
    {
      backend,
      system,
      cudaVersion ? null,
      ptxasVersion ? cudaVersion,
      rocmVersion ? null,
      xpuVersion ? null,
      ...
    }:
    let
      backendVersion =
        if backend == "cuda" then
          "${cudaVersion}-ptxas${ptxasVersion}"
        else if backend == "rocm" then
          rocmVersion
        else if backend == "xpu" then
          xpuVersion
        else
          "";
    in
    "${system}-${backend}-${backendVersion}";

  # Construct the nixpkgs package set for the given backend versions. The
  # package set does not depend on the Torch version.
  mkPkgs =
    {
      backend,
      system,
      cudaVersion ? null,
      ptxasVersion ? cudaVersion,
      rocmVersion ? null,
      xpuVersion ? null,
      ...
    }:
    let
      backendOverlay =
        if backend == "cpu" then
          [ ]
        else if backend == "cuda" then
          [ (overlayForCudaVersion cudaVersion ptxasVersion) ]
        else if backend == "rocm" then
          [ (overlayForRocmVersion rocmVersion) ]
        else if backend == "tpu" then
          [ ]
        else if backend == "metal" then
          [ ]
        else if backend == "xpu" then
          [ (overlayForXpuVersion xpuVersion) ]
        else
          throw "No compute framework set in Torch version";
      config = backendConfig.${backend} or (throw "No backend config for ${backend}");
    in
    import nixpkgs {
      inherit config system;
      overlays = [
        overlay
        rust-overlay.overlays.default
      ]
      ++ backendOverlay
      ++ [ noTorchOverlay ];
    };

  # Construct a build set for the given build config, using `pkgs`.
  mkBuildSet =
    pkgs:
    buildConfig@{
      torchVersion,
      bundleBuild ? false,
      ...
    }:
    let
      torch = pkgs.python3.pkgs."torch-bin_${flattenVersion torchVersion}";

      # Override the `torch` argument of a package if present.
      overrideTorch =
        pkg: if (lib.functionArgs pkg.override) ? torch then pkg.override { inherit torch; } else pkg;

      # python3 is not shared between build sets, so requires build
      # set-specific evaluation. For this reason, it is best to avoid
      # python3.pkgs unless the full set is needed. For derivations
      # that need the build set-specific Torch, use `torch` or
      # `overrideTorch`.
      python3 = pkgs.python3.override {
        self = python3;
        packageOverrides = python-self: python-super: {
          inherit torch;
        };
      };

      extension = pkgs.callPackage ./extension { inherit torch overrideTorch; };

      variants = import ./variants {
        inherit lib buildConfig;
      };
    in
    {
      inherit
        buildConfig
        extension
        pkgs
        python3
        torch
        overrideTorch
        bundleBuild
        variants
        ;
    };
}
