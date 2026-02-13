{
  lib,
  build,
  system,

  writeScriptBin,
  runCommand,

  path,
  buildSets,
  rev ? null,
  self ? null,

  doGetKernelCheck,
  pythonCheckInputs,
  pythonNativeCheckInputs,
}:

let
  supportedFormat = ''
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
  '';
  flakeRev =
    if self != null then
      self.shortRev or self.dirtyShortRev or (builtins.warn ''
        Kernel is not in a git repository, this will create a non-reproducible build.
        This will not be supported in the future.
      '' self.lastModifiedDate)
    else if rev != null then
      builtins.warn "`rev` argument of `genKernelFlakeOutputs` is deprecated, pass `self` as follows:\n\n${supportedFormat}" rev
    else
      throw "Flake's `self` must be passed to `genKernelFlakeOutputs` as follows:\n\n${supportedFormat}";

  revUnderscored = builtins.replaceStrings [ "-" ] [ "_" ] flakeRev;

  applicableBuildSets = build.applicableBuildSets { inherit path buildSets; };

  buildConfigBackend =
    buildConfig:
    if buildConfig.cpu or false then
      "cpu"
    else if buildConfig ? cudaVersion then
      "cuda"
    else if buildConfig ? rocmVersion then
      "rocm"
    else if buildConfig ? xpuVersion then
      "xpu"
    else if buildConfig.metal or false then
      "metal"
    else
      throw "Cannot determine framework for build set";

  # For picking a default shell, etc. we want to use the following logic:
  #
  # - Prefer bundle builds over non-bundle builds.
  # - Prefer CUDA over other frameworks.
  # - Prefer newer Torch versions over older.
  # - Prefer older frameworks over newer (best compatibility).

  # Enrich the build configs with generic attributes for framework
  # order/version. Also make bundleBuild attr explicit.
  addSortOrder = map (
    set:
    let
      inherit (set) buildConfig;
    in
    set
    // {
      buildConfig =

        buildConfig // {
          bundleBuild = buildConfig.bundleBuild or false;
          framework = buildConfigBackend buildConfig;
          frameworkOrder = if buildConfig ? cudaVersion then 0 else 1;
          frameworkVersion =
            buildConfig.cudaVersion or buildConfig.rocmVersion or buildConfig.xpuVersion or "0.0";
        };
    }
  );
  configCompare =
    setA: setB:
    let
      a = setA.buildConfig;
      b = setB.buildConfig;
    in
    if a.bundleBuild != b.bundleBuild then
      a.bundleBuild
    else if a.frameworkOrder != b.frameworkOrder then
      a.frameworkOrder < b.frameworkOrder
    else if a.torchVersion != b.torchVersion then
      builtins.compareVersions a.torchVersion b.torchVersion > 0
    else
      builtins.compareVersions a.frameworkVersion b.frameworkVersion < 0;
  buildSetsSorted = lib.sort configCompare (addSortOrder applicableBuildSets);
  bestBuildSet =
    if buildSetsSorted == [ ] then
      throw "No build variant is compatible with this system"
    else
      builtins.head buildSetsSorted;
  shellTorch = bestBuildSet.torch.variant;
  # We need a package set for some outputs (e.g. kernels and build-and-upload),
  # even when there is no applicable build set.
  pkgs =
    let
      sorted = lib.sort configCompare (addSortOrder buildSets);
    in
    if sorted == [ ] then
      throw "No build set is available for this system"
    else
      (builtins.head sorted).pkgs;
  headOrEmpty = l: if l == [ ] then [ ] else [ (builtins.head l) ];
in
{
  devShells = rec {
    default = devShells.${shellTorch};
    test = testShells.${shellTorch};
    devShells = build.mkTorchDevShells {
      inherit
        path
        doGetKernelCheck
        pythonCheckInputs
        pythonNativeCheckInputs
        ;
      buildSets = applicableBuildSets;
      rev = revUnderscored;
    };
    testShells = build.torchExtensionShells {
      inherit
        path
        doGetKernelCheck
        pythonCheckInputs
        pythonNativeCheckInputs
        ;
      buildSets = applicableBuildSets;
      rev = revUnderscored;
    };
  };
  packages =
    let
      bundle = build.mkTorchExtensionBundle {
        inherit path doGetKernelCheck;
        buildSets = applicableBuildSets;
        rev = revUnderscored;
      };
      ciTests = build.mkCiTests {
        inherit path doGetKernelCheck pythonCheckInputs;
        buildSets = applicableBuildSets;
        rev = revUnderscored;
      };
    in
    {
      inherit bundle ciTests;

      # Bundles by backend.
      backendBundle =
        let
          backends = lib.unique (map (set: buildConfigBackend set.buildConfig) applicableBuildSets);
        in
        builtins.listToAttrs (
          builtins.map (backend: {
            name = backend;
            value = build.mkTorchExtensionBundle {
              inherit path doGetKernelCheck;
              buildSets = builtins.filter (
                set: buildConfigBackend set.buildConfig == backend
              ) applicableBuildSets;
              rev = revUnderscored;
            };
          }) backends
        );

      default = bundle;

      build-and-copy = writeScriptBin "build-and-copy" ''
        #!/usr/bin/env bash
        set -euo pipefail

        if [ ! -d build ]; then
          mkdir build
        fi

        for build_variant in ${bundle}/*; do
          build_variant=$(basename $build_variant)
          if [ -e build/$build_variant ]; then
            rm -rf build/$build_variant
          fi

          cp -r ${bundle}/$build_variant build/
        done

        chmod -R +w build
      '';

      build-and-upload =
        let
          buildToml = build.readBuildConfig path;
          repo_id = lib.attrByPath [
            "general"
            "hub"
            "repo-id"
          ] "kernels-community/${buildToml.general.name}" buildToml;
          branch = lib.attrByPath [ "general" "hub" "branch" ] null buildToml;
          branchOpt = lib.optionalString (branch != null) "--branch ${branch}";
          # `kernels upload` fails when there are no build variants to upload.
          # However, we do not want this command to error out in that case, so
          # only insert the upload command when there is something to upload.
          uploadStr = lib.optionalString (applicableBuildSets != [ ]) ''
            ${pkgs.python3.pkgs.kernels}/bin/kernels upload --repo-id ${repo_id} ${branchOpt} ${bundle}
          '';
        in
        writeScriptBin "build-and-upload" ''
          #!/usr/bin/env bash
          set -euo pipefail
          ${uploadStr}
        '';

      ci =
        let
          setsWithFramework =
            framework: builtins.filter (set: set.buildConfig.framework == framework) buildSetsSorted;
          # It is too costly to build all variants in CI, so we just build one per framework.
          onePerFramework =
            (headOrEmpty (setsWithFramework "cpu"))
            ++ (headOrEmpty (setsWithFramework "cuda"))
            ++ (headOrEmpty (setsWithFramework "metal"))
            ++ (headOrEmpty (setsWithFramework "rocm"))
            ++ (headOrEmpty (setsWithFramework "xpu"));
        in
        build.mkTorchExtensionBundle {
          inherit path doGetKernelCheck;
          buildSets = onePerFramework;
          rev = revUnderscored;
        };

      ci-test = ciTests.${shellTorch};

      kernels =
        pkgs.python3.withPackages (
          ps: with ps; [
            kernel-abi-check
            kernels
          ]
        )
        // {
          meta.mainProgram = "kernels";
        };

      redistributable = build.mkDistTorchExtensions {
        inherit path doGetKernelCheck;
        bundleOnly = false;
        rev = revUnderscored;
        buildSets = applicableBuildSets;
      };
    };
}
