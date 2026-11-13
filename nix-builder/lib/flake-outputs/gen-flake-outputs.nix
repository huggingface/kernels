{
  lib,

  writeScriptBin,

  # Filtered set of build sets that are compatible with she
  # system + kernel/sdk.
  applicableBuildSets,

  # All buil sets. Used to access utilities when there are no applicable
  # build test.
  buildSets,

  # Upload repo-id.
  repoId,

  # Upload branch.
  branch ? null,

  # Variant naming (build set -> variant name).
  variantOf,

  # Arch variant naming (build set -> variant name).
  archVariantOf,

  # Bundle builder.
  mkBundle,

  # Redistributable buildset builder.
  mkRedistributable,

  # Optional builders. The corresponding outputs are omitted when `null`.
  mkDevShells ? null,
  mkTestShells ? null,
  mkCiTests ? null,

  # Additional packages, `{ pkgs, overrideTorch }: attrs` to add to the
  # flake packages.
  extraPackages ? _: { },
}:

let
  # For picking a default shell, etc. we want to use the following logic:
  #
  # - Prefer bundle builds over non-bundle builds.
  # - Prefer CUDA over other frameworks.
  # - Prefer newer Torch versions over older.
  # - Prefer older frameworks over newer (best compatibility).

  backendOrder = {
    cuda = 0;
    rocm = 1;
    xpu = 2;
    metal = 3;
  };

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
          frameworkOrder = backendOrder.${buildConfig.backend} or 4;
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
  bestVariant = archVariantOf bestBuildSet;
  # We need a package set for some outputs (e.g. kernels and build-and-upload),
  # even when there is no applicable build set.
  fallbackBuildSet =
    let
      sorted = lib.sort configCompare (addSortOrder buildSets);
    in
    if sorted == [ ] then throw "No build set is available for this system" else builtins.head sorted;
  inherit (fallbackBuildSet) pkgs overrideTorch;
  backendOf = set: set.buildConfig.backend;
  backendSets = backend: sets: builtins.filter (set: backendOf set == backend) sets;
  headOrEmpty = l: if l == [ ] then [ ] else [ (builtins.head l) ];
in
{
  devShells =
    let
      devShells = mkDevShells applicableBuildSets;
      testShells = mkTestShells applicableBuildSets;
    in
    lib.optionalAttrs (mkDevShells != null) {
      default = devShells.${bestVariant};
      inherit devShells;
    }
    // lib.optionalAttrs (mkTestShells != null) {
      test = testShells.${bestVariant};
      inherit testShells;
    };
  packages =
    let
      bundle = mkBundle applicableBuildSets;
      ciTests = mkCiTests applicableBuildSets;
    in
    {
      inherit bundle;

      # Bundles by backend.
      backendBundle =
        let
          backends = lib.unique (map backendOf applicableBuildSets);
        in
        builtins.listToAttrs (
          builtins.map (backend: {
            name = backend;
            value = mkBundle (backendSets backend applicableBuildSets);
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
          branchOpt = lib.optionalString (branch != null) "--branch ${branch}";
          # `kernels upload` fails when there are no build variants to upload.
          # However, we do not want this command to error out in that case, so
          # only insert the upload command when there is something to upload.
          uploadStr = lib.optionalString (applicableBuildSets != [ ]) ''
            ${pkgs.kernel-builder}/bin/kernel-builder upload --repo-id ${repoId} ${branchOpt} ${bundle}
          '';
        in
        writeScriptBin "build-and-upload" ''
          #!/usr/bin/env bash
          set -euo pipefail
          ${uploadStr}
        '';

      # It is too costly to build all variants in CI, so we just build one per
      # backend.
      ci = mkBundle (
        lib.concatMap (backend: headOrEmpty (backendSets backend buildSetsSorted)) (
          lib.sort lib.lessThan (lib.unique (map backendOf buildSetsSorted))
        )
      );

      backendCi =
        let
          backends = lib.unique (map backendOf buildSetsSorted);
        in
        builtins.listToAttrs (
          builtins.map (backend: {
            name = backend;
            value = mkBundle (headOrEmpty (backendSets backend buildSetsSorted));
          }) backends
        );

      redistributable = mkRedistributable applicableBuildSets;

      # We do not really have a great place to put these. We need them to be
      # keyed by system, so we cannot put thin in the `lib` output.

      variants = lib.unique (map variantOf applicableBuildSets);

      archVariants = lib.unique (map archVariantOf applicableBuildSets);
    }
    // lib.optionalAttrs (mkCiTests != null) {
      inherit ciTests;
      ci-test = ciTests.${bestVariant};
    }
    // extraPackages { inherit pkgs overrideTorch; };
}
