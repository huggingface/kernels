{
  lib,
  pkgs,

  # Every `buildSets` argument is a list of build sets. Each build set is
  # a attrset of the form
  #
  #     { pkgs = <nixpkgs>, torch = <torch drv> }
  #
  # The Torch derivation is built as-is. So e.g. the ABI version should
  # already be set.
}:

let
  readKernelConfig = import ./kernel-config.nix { inherit lib; };
  supportedCudaCapabilities = builtins.fromJSON (
    builtins.readFile ../../kernel-builder/src/cuda_supported_archs.json
  );
in
rec {
  # Source set function to create a fileset for a path
  mkSourceSet = import ./source-set.nix { inherit lib; };

  # Filter buildsets that are applicable to a given kernel build config.
  filterApplicableBuildSets =
    kernelConfig: buildSets:
    let
      minCuda = kernelConfig.toml.general.cuda.minver or "11.8";
      maxCuda = kernelConfig.toml.general.cuda.maxver or "99.9";
      minTorch = kernelConfig.toml.torch.minver or "2.0";
      maxTorch = kernelConfig.toml.torch.maxver or "99.9";
      versionBetween =
        minver: maxver: ver:
        builtins.compareVersions ver minver >= 0 && builtins.compareVersions ver maxver <= 0;
      supportedBuildSet =
        buildSet:
        let
          backendSupported = kernelConfig.backends.${buildSet.buildConfig.backend};
          frameworkSupported = !kernelConfig.isTvmFfi || (buildSet.buildConfig ? tvmFfiVersion);
          cudaVersionSupported =
            buildSet.buildConfig.backend != "cuda"
            || versionBetween minCuda maxCuda buildSet.buildConfig.cudaVersion;
          torchVersionSupported = versionBetween minTorch maxTorch buildSet.buildConfig.torchVersion;
        in
        backendSupported && cudaVersionSupported && frameworkSupported && torchVersionSupported;
    in
    builtins.filter supportedBuildSet buildSets;

  applicableBuildSets =
    { path, buildSets }: filterApplicableBuildSets (readKernelConfig path) buildSets;

  # Build a single extension.
  mkExtension =
    {
      buildConfig,
      extension,
      pkgs,
      torch,
      bundleBuild,
      variants,
    }:
    {
      kernelConfig,
      path,
      rev,
      doGetKernelCheck,
      stripRPath ? false,
    }:
    let
      inherit (lib) fileset;
      kernels = lib.filterAttrs (_: kernel: buildConfig.backend == kernel.backend) (
        kernelConfig.toml.kernel or { }
      );
      extraDeps =
        let
          inherit (import ./deps.nix { inherit lib pkgs torch; }) resolveCppDeps;
          kernelDeps = lib.unique (lib.flatten (lib.mapAttrsToList (_: kernel: kernel.depends) kernels));
        in
        resolveCppDeps kernelDeps;

      # Use the mkSourceSet function to get the source
      src = mkSourceSet path;

      # Set number of threads to the largest number of capabilities.
      listMax = lib.foldl' lib.max 1;
      nvccThreads = listMax (
        lib.mapAttrsToList (
          _: kernel: builtins.length (kernel.cuda-capabilities or supportedCudaCapabilities)
        ) kernelConfig.toml.kernel
      );
      pythonDeps = (kernelConfig.toml.general.python-depends or [ ]);
      backendPythonDeps =
        lib.attrByPath [ buildConfig.backend "python-depends" ] [ ]
          kernelConfig.toml.general;
    in
    if !kernelConfig.kernelBackends.${buildConfig.backend} then
      # No compiled kernel files? Treat it as a noarch package.

      extension.mkTorchNoArchExtension {
        inherit
          buildConfig
          src
          rev
          doGetKernelCheck
          pythonDeps
          backendPythonDeps
          ;
        kernelName = kernelConfig.name;
      }
    else if kernelConfig.isTvmFfi then
      extension.mkTvmFfiExtension {
        inherit
          buildConfig
          doGetKernelCheck
          extraDeps
          nvccThreads
          src
          stripRPath
          rev
          pythonDeps
          backendPythonDeps
          ;

        kernelName = kernelConfig.name;
        doAbiCheck = true;
      }
    else
      extension.mkTorchExtension {
        inherit
          buildConfig
          doGetKernelCheck
          extraDeps
          nvccThreads
          src
          stripRPath
          rev
          pythonDeps
          backendPythonDeps
          ;

        kernelName = kernelConfig.name;
        doAbiCheck = true;
      };

  # Build multiple Torch extensions.
  mkDistTorchExtensions =
    {
      path,
      rev,
      doGetKernelCheck,
      bundleOnly,
      buildSets,
    }:
    let
      kernelConfig = readKernelConfig path;
      extensionForTorch =
        { path, rev }:
        buildSet: rec {
          name = buildSet.variants.kernelVariant kernelConfig;
          value = mkExtension buildSet {
            inherit
              path
              kernelConfig
              rev
              doGetKernelCheck
              ;
            stripRPath = true;
          };
        };
      applicableBuildSets' =
        if bundleOnly then builtins.filter (buildSet: buildSet.bundleBuild) buildSets else buildSets;
    in
    builtins.listToAttrs (lib.map (extensionForTorch { inherit path rev; }) applicableBuildSets');

  mkExtensionBundle =
    {
      path,
      rev,
      doGetKernelCheck,
      buildSets,
    }:
    let
      extensions = mkDistTorchExtensions {
        inherit
          buildSets
          path
          rev
          doGetKernelCheck
          ;
        bundleOnly = true;
      };
      benchmarksPath = path + "/benchmarks";
      hasBenchmarks = builtins.pathExists benchmarksPath;
      benchmarks =
        with lib.fileset;
        toSource {
          root = path;
          fileset = maybeMissing benchmarksPath;
        };
      contents =
        builtins.map (pkg: toString pkg) (builtins.attrValues extensions)
        ++ lib.optionals hasBenchmarks [ (toString benchmarks) ];
    in
    import ./join-paths {
      inherit pkgs contents;
      name = "torch-ext-bundle";
      # Fill card iff we build any variants.
      postInstall = lib.optionalString (buildSets != [ ]) ''
        if [ -f "${path}/CARD.md" ]; then
          ${(builtins.head buildSets).pkgs.kernel-builder}/bin/kernel-builder fill-card ${path} $out/CARD.md
        fi
      '';
    };

  # Get a development shell with the extension in PYTHONPATH. Handy
  # for running tests.
  torchExtensionShells =
    {
      path,
      rev,
      buildSets,
      doGetKernelCheck,
      pythonCheckInputs,
      pythonNativeCheckInputs,
    }:
    let
      kernelConfig = readKernelConfig path;
      shellForBuildSet =
        { path, rev }:
        buildSet:
        let
          pkgs = buildSet.pkgs;
          rocmSupport = pkgs.config.rocmSupport or false;
          mkShell = pkgs.mkShell.override { inherit (buildSet.extension) stdenv; };
          extension = mkExtension buildSet {
            inherit
              path
              kernelConfig
              rev
              doGetKernelCheck
              ;
          };
        in
        {
          name = buildSet.variants.kernelArchVariant kernelConfig;
          value = mkShell {
            nativeBuildInputs = with pkgs; pythonNativeCheckInputs python3.pkgs;

            buildInputs = with pkgs; [
              (python3.withPackages (
                ps:
                with ps;
                extension.dependencies
                ++ pythonCheckInputs ps
                ++ [
                  buildSet.torch
                  pytest
                ]
                ++ pythonCheckInputs ps
                ++ lib.optionals kernelConfig.isTvmFfi [
                  jax
                  jax-tvm-ffi
                  numpy
                  tvm-ffi
                ]
              ))
            ];
            shellHook = ''
              # This is run from `nix develop`, which provides the existing
              # environment. We clear the LD_LIBRARY_PATH and PYTHONPATH to
              # make testing as pure as possible.
              unset LD_LIBRARY_PATH
              export PYTHONPATH=${extension}/${buildSet.variants.kernelVariant kernelConfig}
            '';
          };
        };
    in
    builtins.listToAttrs (lib.map (shellForBuildSet { inherit path rev; }) buildSets);

  # Kernel CI test runners.
  mkCiTests =
    {
      path,
      rev,
      buildSets,
      doGetKernelCheck,
      pythonCheckInputs,
    }:
    let
      kernelConfig = readKernelConfig path;
      runnerForBuildSet =
        { path, rev }:
        buildSet:
        let
          pkgs = buildSet.pkgs;
          rocmSupport = pkgs.config.rocmSupport or false;
          mkShell = pkgs.mkShell.override { inherit (buildSet.extension) stdenv; };
          extension = mkExtension buildSet {
            inherit
              path
              kernelConfig
              rev
              doGetKernelCheck
              ;
          };
          testPython =
            with pkgs;
            python3.withPackages (
              ps:
              with ps;
              extension.dependencies
              ++ [
                buildSet.torch
                pytest
              ]
              ++ pythonCheckInputs ps
            );
        in
        {
          name = buildSet.variants.kernelArchVariant kernelConfig;
          value =
            with pkgs;
            pkgs.writeShellScriptBin "ci-test" ''
              if [ -d ${extension.src}/tests ]; then
                unset LD_LIBRARY_PATH
                export PYTHONPATH=${extension}/${buildSet.variants.kernelVariant kernelConfig}
                # Accept exit code 5: no tests are selected
                ${testPython}/bin/python3 -m pytest ${extension.src}/tests -m kernels_ci -p no:cacheprovider || test $? -eq 5
              fi
            '';
        };
    in
    builtins.listToAttrs (lib.map (runnerForBuildSet { inherit path rev; }) buildSets);

  mkDevShells =
    {
      path,
      rev,
      buildSets,
      doGetKernelCheck,
      pythonCheckInputs,
      pythonNativeCheckInputs,
    }:
    let
      kernelConfig = readKernelConfig path;
      shellForBuildSet =
        buildSet:
        let
          pkgs = buildSet.pkgs;
          rocmSupport = pkgs.config.rocmSupport or false;
          xpuSupport = pkgs.config.xpuSupport or false;
          mkShell = pkgs.mkShell.override { inherit (buildSet.extension) stdenv; };
          extension = mkExtension buildSet {
            inherit
              path
              kernelConfig
              rev
              doGetKernelCheck
              ;
          };
          python = (
            pkgs.python3.withPackages (
              ps:
              with ps;
              extension.dependencies
              ++ pythonCheckInputs ps
              ++ [
                buildSet.torch
                kernels
                pip
                pytest
              ]
              ++ lib.optionals kernelConfig.isTvmFfi [
                jax
                jax-tvm-ffi
                numpy
                tvm-ffi
              ]
            )
          );
        in
        {
          name = buildSet.variants.kernelArchVariant kernelConfig;
          value = mkShell rec {
            nativeBuildInputs =
              with pkgs;
              [
                kernel-builder
                kernel-abi-check
              ]
              ++ (pythonNativeCheckInputs python3.pkgs);
            buildInputs = [ python ];
            inputsFrom = [ extension ];
            env = lib.optionalAttrs rocmSupport {
              PYTORCH_ROCM_ARCH = lib.concatStringsSep ";" buildSet.torch.rocmArchs;
              HIP_PATH = pkgs.rocmPackages.clr;
            };

            venvDir = "./.venv";

            # We don't use venvShellHook because we want to use our wrapped
            # Python interpreter.
            shellHook = ''
              if [ -d "${venvDir}" ]; then
                echo "Skipping venv creation, '${venvDir}' already exists"
              else
                echo "Creating new venv environment in path: '${venvDir}'"
                ${python}/bin/python -m venv --system-site-packages "${venvDir}"
              fi
              source "${venvDir}/bin/activate"
              unset LD_LIBRARY_PATH
            '';
          };
        };
    in
    builtins.listToAttrs (lib.map shellForBuildSet buildSets);
}
