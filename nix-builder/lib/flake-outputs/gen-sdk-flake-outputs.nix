{
  lib,
  writeScriptBin,
  writeShellScriptBin,
  runCommand,

  # Build sets.
  buildSets,

  self,

  # Hub repository to upload to (and optionally the branch).
  repoId,
  branch ? null,

  # Python version and ABI that the SDK is built for, e.g.
  # `{ version = "3.12"; abi = "abi3"; }`.
  python,

  # Directory with tests, run with pytest. Tests marked with `kernels_ci`
  # are run by `ci-test`.
  tests ? null,

  # Build one variant of the SDK, called with `{ buildSet, variant, rev,
  # buildSets }` where `buildSets`. Returns a derivation.
  mkVariant,

  # Additional packages to include in the flake's packages.
  extraPackages ? _: { },
}:

let
  rev = lib.replaceStrings [ "-" ] [ "_" ] (
    self.shortRev or self.dirtyShortRev or self.lastModifiedDate
  );

  # Ensure that the Python of the build sets can build/load SDKs for the
  # Python version of the SDK.
  checkPython =
    buildSet:
    lib.assertMsg
      (!(lib.hasPrefix "abi3" python.abi) || lib.versionAtLeast buildSet.python3.version python.version)
      "Build set Python ${buildSet.python3.version} cannot load an SDK for Python ${python.version} (${python.abi})";

  applicableBuildSets = lib.filter checkPython buildSets;

  variantOf = buildSet: buildSet.variants.sdkVariant python;

  variantSets = lib.listToAttrs (
    map (buildSet: {
      name = variantOf buildSet;
      value = mkVariant {
        inherit buildSet rev;
        variant = variantOf buildSet;
        buildSets = applicableBuildSets;
      };
    }) applicableBuildSets
  );
  variantDrvs = lib.mapAttrs (_: set: set.kernel) variantSets;

  variantsOf = sets: lib.genAttrs (map variantOf sets) (variant: variantSets.${variant});

  testPython =
    buildSet:
    buildSet.python3.withPackages (ps: [
      buildSet.torch
      (buildSet.overrideTorch buildSet.python3.pkgs.kernels)
      ps.pytest
    ]);

  localKernels = variant: "${repoId}=${variantDrvs.${variant}}";
in
lib.optionalAttrs (applicableBuildSets != [ ]) (
  import ./gen-flake-outputs.nix {
    inherit
      lib
      writeScriptBin
      applicableBuildSets
      repoId
      branch
      variantOf
      ;
    buildSets = applicableBuildSets;
    archVariantOf = variantOf;

    mkBundle =
      sets:
      runCommand "sdk-bundle"
        {
          preferLocalBuild = true;
          allowSubstitutes = false;
        }
        ''
          mkdir -p $out
          ${lib.concatMapStringsSep "\n" (set: "cp -r ${variantDrvs.${variantOf set}}/* $out/") sets}
        '';

    mkRedistributable = sets: lib.mapAttrs (_: set: set.kernel) (variantsOf sets);

    mkTestShells =
      sets:
      lib.listToAttrs (
        map (buildSet: {
          name = variantOf buildSet;
          value = buildSet.pkgs.mkShell {
            buildInputs = [ (testPython buildSet) ];
            shellHook = ''
              # This is run from `nix develop`, which provides the existing
              # environment. We clear the LD_LIBRARY_PATH to make testing as
              # pure as possible.
              unset LD_LIBRARY_PATH
              export PYTHONPATH=${variantDrvs.${variantOf buildSet}}/${variantOf buildSet}
              export LOCAL_KERNELS="${localKernels (variantOf buildSet)}"
            '';
          };
        }) sets
      );

    mkCiTests =
      if tests == null then
        null
      else
        sets:
        lib.listToAttrs (
          map (buildSet: {
            name = variantOf buildSet;
            value = writeShellScriptBin "ci-test" ''
              unset LD_LIBRARY_PATH
              export LOCAL_KERNELS="${localKernels (variantOf buildSet)}"
              # Accept exit code 5: no tests are selected
              ${testPython buildSet}/bin/python3 -m pytest ${tests} -m kernels_ci -p no:cacheprovider || test $? -eq 5
            '';
          }) sets
        );

    extraPackages =
      args:
      extraPackages (
        args
        // {
          inherit variantSets;
          buildSets = applicableBuildSets;
        }
      );
  }
)
