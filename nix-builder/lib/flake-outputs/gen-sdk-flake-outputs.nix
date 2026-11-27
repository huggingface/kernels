{
  lib,
  writeScriptBin,
  writeShellScriptBin,
  runCommand,

  # Build sets.
  buildSets,

  self,

  # Git provenance (`{ commit, dirty }` or `null`) of the `kernels` flake.
  builderProvenance ? null,

  # Hub repository to upload to (and optionally the branch).
  repoId,
  branch ? null,

  # Name of the SDK (e.g. `flydsl`).
  name,

  # Python version and ABI that the SDK is built for, e.g.
  # `{ version = "3.12"; abi = "abi3"; }`. Only `abi3` is currently
  # supported. Set to `null` for a noarch SDK.
  python ? null,

  # Whether the SDK has backend variants.
  useBackend ? true,

  # Make the tests for one variant of the SDK. This is a function to allow
  # the SDK flake to pull the tests from the source, while also e.g. replacing
  # imports by `get_kernel` calls.
  mkTests ? null,

  # Build one variant of the SDK, called with `{ buildSet, variant, rev,
  # buildSets }` where `buildSets`. Returns a derivation.
  mkVariant,
}:

let
  rev = lib.replaceStrings [ "-" ] [ "_" ] (
    self.shortRev or self.dirtyShortRev or self.lastModifiedDate
  );

  noarch = python == null;

  # Ensure that the Python of the build sets can build/load SDKs for the
  # Python version of the SDK.
  checkPython =
    buildSet:
    lib.assertMsg
      (!(lib.hasPrefix "abi3" python.abi) || lib.versionAtLeast buildSet.python3.version python.version)
      "Build set Python ${buildSet.python3.version} cannot load an SDK for Python ${python.version} (${python.abi})";

  variantOf =
    buildSet:
    if !noarch then
      if useBackend then
        buildSet.variants.sdkVariant python
      else
        buildSet.variants.sdkVariantNoBackend python
    else if !useBackend then
      buildSet.variants.sdkNoarchVariantNoBackend
    else
      buildSet.variants.sdkNoarchVariant;

  # Filter buildsets that can build for the given Python ABI version for
  # arch SDKs. For noarch SDKs, all build sets are applicable.
  applicableBuildSets = if noarch then buildSets else lib.filter checkPython buildSets;

  # SDK source provenance that is added to the metadata.
  provenance = {
    kernel = import ./flake-provenance.nix { inherit lib; } self;
    builder = builderProvenance;
  };

  # Check and finalize a variant. Copy the build output and run our standard
  # kernel hooks on the copy.
  checkVariant =
    {
      buildSet,
      variant,
      sdk,
    }:
    let
      inherit (buildSet)
        pkgs
        python3
        torch
        overrideTorch
        ;

      # Torch is not used by SDKs, it is only needed in the environment of the
      # load check, which provides the backend libraries (e.g. ROCm).
      loadCheckPython = python3.withPackages (_: [ torch ]);
      kernels = overrideTorch python3.pkgs.kernels;
    in
    assert lib.assertMsg (
      noarch || python.abi == "abi3"
    ) "Only abi3 SDKs are supported, got ABI `${python.abi}` for `${name}`";
    pkgs.stdenv.mkDerivation {
      name = "${name}-${variant}";

      dontUnpack = true;

      # SDKs can contain multiple libraries that depend on each other, using
      # rpaths relative to the library origin. In such cases, we want to keep
      # the origin-rlative rpaths, but remove anyother.
      stripRPathKeepOrigin = true;

      nativeBuildInputs = [
        pkgs.strip-rpath-hook
        # Order matters: remove bytecode, then hash.
        pkgs.remove-bytecode-hook
        pkgs.hash-kernel-hook
        pkgs.kernel-builder
        (pkgs.get-kernel-check.override {
          python3 = loadCheckPython;
          inherit kernels;
        })
        (pkgs.generate-symbols-hook.override {
          python3 = loadCheckPython;
          inherit kernels;
        })
      ];

      moduleName = name;
      kernelDeps = pkgs.writeText "kernel-deps" "[]";
      inherit variant;

      doInstallCheck = true;
      doAbiCheck = !noarch;
      pythonAbiVersion = if noarch then "" else python.version;

      # Provenance is added by the builder, replacing any that the SDK wrote.
      provenance = pkgs.writeText "provenance.json" (builtins.toJSON provenance);

      installPhase = ''
        runHook preInstall

        mkdir -p $out
        cp -r ${sdk}/. $out/
        chmod -R u+w $out

        runHook postInstall
      '';
    };

  variantDrvs = lib.listToAttrs (
    map (
      buildSet:
      let
        variant = variantOf buildSet;
      in
      {
        name = variant;
        value = checkVariant {
          inherit buildSet variant;
          sdk = mkVariant {
            inherit buildSet variant rev;
            buildSets = applicableBuildSets;
          };
        };
      }
    ) applicableBuildSets
  );

  testPython =
    buildSet:
    buildSet.python3.withPackages (ps: [
      buildSet.torch
      (buildSet.overrideTorch buildSet.python3.pkgs.kernels)
      ps.pytest
    ]);

  localKernels = variant: "${repoId}=${variantDrvs.${variant}}";
in
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

  mkRedistributable = sets: lib.genAttrs (map variantOf sets) (variant: variantDrvs.${variant});

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
    if mkTests == null then
      null
    else
      sets:
      lib.listToAttrs (
        map (
          buildSet:
          let
            variant = variantOf buildSet;
            tests = mkTests {
              inherit buildSet variant rev;
              buildSets = applicableBuildSets;
            };
          in
          {
            name = variant;
            value = writeShellScriptBin "ci-test" ''
              unset LD_LIBRARY_PATH
              export LOCAL_KERNELS="${localKernels variant}"
              # Accept exit code 5: no tests are selected
              ${testPython buildSet}/bin/python3 -m pytest ${tests} -m kernels_ci -p no:cacheprovider || test $? -eq 5
            '';
          }
        ) sets
      );
}
