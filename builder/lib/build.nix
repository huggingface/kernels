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
  supportedCudaCapabilities = builtins.fromJSON (
    builtins.readFile ../../build2cmake/src/cuda_supported_archs.json
  );
in
rec {
  readToml = path: builtins.fromTOML (builtins.readFile path);

  validateBuildConfig =
    buildToml:
    let
      hasBackends = buildToml.general ? backends;
      kernels = lib.attrValues (buildToml.kernel or { });

    in
    assert lib.assertMsg hasBackends ''
      build.toml seems to be of an older version, update it with:
            nix run github:huggingface/kernel-builder#build2cmake update-build build.toml'';
    buildToml;

  # Backends supported by the kernel.
  backends =
    buildToml:
    let
      init = {
        cpu = false;
        cuda = false;
        metal = false;
        rocm = false;
        xpu = false;
      };
    in
    lib.foldl (backends: backend: backends // { ${backend} = true; }) init (buildToml.general.backends);

  # Backends for which there is a native (compiled kernel).
  kernelBackends =
    buildToml:
    let
      kernels = lib.attrValues (buildToml.kernel or { });
      kernelBackend = kernel: kernel.backend;
      init = {
        cpu = false;
        cuda = false;
        metal = false;
        rocm = false;
        xpu = false;
      };
    in
    lib.foldl (backends: kernel: backends // { ${kernelBackend kernel} = true; }) init kernels;

  readBuildConfig = path: validateBuildConfig (readToml (path + "/build.toml"));

  srcFilter =
    src: name: type:
    type == "directory" || lib.any (suffix: lib.hasSuffix suffix name) src;

  # Source set function to create a fileset for a path
  mkSourceSet = import ./source-set.nix { inherit lib; };

  # Filter buildsets that are applicable to a given kernel build config.
  filterApplicableBuildSets =
    buildToml: buildSets:
    let
      backends' = backends buildToml;
      minCuda = buildToml.general.cuda.minver or "11.8";
      maxCuda = buildToml.general.cuda.maxver or "99.9";
      minTorch = buildToml.torch.minver or "2.0";
      maxTorch = buildToml.torch.maxver or "99.9";
      versionBetween =
        minver: maxver: ver:
        builtins.compareVersions ver minver >= 0 && builtins.compareVersions ver maxver <= 0;
      supportedBuildSet =
        buildSet:
        let
          backendSupported = backends'.${buildSet.buildConfig.backend};
          cudaVersionSupported =
            buildSet.buildConfig.backend != "cuda"
            || versionBetween minCuda maxCuda buildSet.pkgs.cudaPackages.cudaMajorMinorVersion;
          torchVersionParts = lib.splitString "." buildSet.torch.version;
          torchMajorMinor = lib.concatStringsSep "." (lib.take 2 torchVersionParts);
          torchVersionSupported = versionBetween minTorch maxTorch torchMajorMinor;
        in
        backendSupported && cudaVersionSupported && torchVersionSupported;
    in
    builtins.filter supportedBuildSet buildSets;

  applicableBuildSets =
    { path, buildSets }: filterApplicableBuildSets (readBuildConfig path) buildSets;

  # Build a single Torch extension.
  mkTorchExtension =
    {
      buildConfig,
      extension,
      pkgs,
      torch,
      bundleBuild,
    }:
    {
      path,
      rev,
      doGetKernelCheck,
      stripRPath ? false,
    }:
    let
      inherit (lib) fileset;
      buildToml = readBuildConfig path;
      kernelBackends' = kernelBackends buildToml;
      kernels = lib.filterAttrs (_: kernel: buildConfig.backend == kernel.backend) (
        buildToml.kernel or { }
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
        ) buildToml.kernel
      );
      pythonDeps = (buildToml.general.python-depends or [ ]);
      backendPythonDeps = lib.attrByPath [ buildConfig.backend "python-depends" ] [ ] buildToml.general;
    in
    if !kernelBackends'.${buildConfig.backend} then
      # No compiled kernel files? Treat it as a noarch package.

      extension.mkNoArchExtension {
        inherit
          buildConfig
          src
          rev
          doGetKernelCheck
          pythonDeps
          backendPythonDeps
          ;
        kernelName = buildToml.general.name;
      }
    else
      extension.mkExtension {
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

        kernelName = buildToml.general.name;
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
      extensionForTorch =
        { path, rev }:
        buildSet: rec {
          name = value.variant;
          value = mkTorchExtension buildSet {
            inherit path rev doGetKernelCheck;
            stripRPath = true;
          };
        };
      applicableBuildSets' =
        if bundleOnly then builtins.filter (buildSet: buildSet.bundleBuild) buildSets else buildSets;
    in
    builtins.listToAttrs (lib.map (extensionForTorch { inherit path rev; }) applicableBuildSets');

  mkTorchExtensionBundle =
    {
      path,
      rev,
      doGetKernelCheck,
      buildSets,
    }:
    let
      # We just need to get any nixpkgs for use by the path join.
      extensions = mkDistTorchExtensions {
        inherit
          buildSets
          path
          rev
          doGetKernelCheck
          ;
        bundleOnly = true;
      };
      buildToml = readBuildConfig path;
      namePaths =
        # TODO: treat kernels without compiled parts differently.
        lib.mapAttrs (name: pkg: toString pkg) extensions;

      # Include benchmarks directory if it exists in the source
      benchmarksPath = path + "/benchmarks";
      hasBenchmarks = builtins.pathExists benchmarksPath;
      allPaths = namePaths // lib.optionalAttrs hasBenchmarks {
        benchmarks = benchmarksPath;
      };
    in
    import ./join-paths {
      inherit pkgs;
      namePaths = allPaths;
      name = "torch-ext-bundle";
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
      shellForBuildSet =
        { path, rev }:
        buildSet:
        let
          pkgs = buildSet.pkgs;
          rocmSupport = pkgs.config.rocmSupport or false;
          mkShell = pkgs.mkShell.override { inherit (buildSet.extension) stdenv; };
          extension = mkTorchExtension buildSet { inherit path rev doGetKernelCheck; };
        in
        {
          name = buildSet.torch.variant;
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
              ))
            ];
            shellHook = ''
              export PYTHONPATH=''${PYTHONPATH}:${extension}
              unset LD_LIBRARY_PATH
            '';
          };
        };
    in
    builtins.listToAttrs (lib.map (shellForBuildSet { inherit path rev; }) buildSets);

  mkTorchDevShells =
    {
      path,
      rev,
      buildSets,
      doGetKernelCheck,
      pythonCheckInputs,
      pythonNativeCheckInputs,
    }:
    let
      shellForBuildSet =
        buildSet:
        let
          pkgs = buildSet.pkgs;
          rocmSupport = pkgs.config.rocmSupport or false;
          xpuSupport = pkgs.config.xpuSupport or false;
          mkShell = pkgs.mkShell.override { inherit (buildSet.extension) stdenv; };
          extension = mkTorchExtension buildSet { inherit path rev doGetKernelCheck; };
          python = (
            pkgs.python3.withPackages (
              ps:
              with ps;
              extension.dependencies
              ++ pythonCheckInputs ps
              ++ [
                buildSet.torch
                pip
                pytest
              ]
            )
          );
        in
        {
          name = buildSet.torch.variant;
          value = mkShell rec {
            nativeBuildInputs =
              with pkgs;
              [
                build2cmake
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
