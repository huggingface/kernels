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
  # `{ version = "3.12"; abi = "abi3"; }`. The ABI is one of: abi3,
  # abi3t, gil, nogil.
  python,

  # Directory with tests, run with pytest. Tests marked with `kernels_ci`
  # are run by `ci-test`.
  tests ? null,

  # Build one variant of the SDK, called with `{ buildSet, variant, rev,
  # buildSets }` where `buildSets`. Returns a derivation.
  mkVariant,
}:

let
  rev = lib.replaceStrings [ "-" ] [ "_" ] (
    self.shortRev or self.dirtyShortRev or self.lastModifiedDate
  );

  stableAbi = lib.hasPrefix "abi3" python.abi;

  isFreeThreaded = python3: lib.hasSuffix "t" python3.executable;

  # Ensure that the Python of the build sets can build/load SDKs for the
  # Python version and ABI of the SDK.
  checkPython =
    buildSet:
    let
      inherit (buildSet) python3;
      message = "Build set Python ${python3.version} (${python3.executable}) cannot load an SDK for Python ${python.version} (${python.abi})";
    in
    lib.assertMsg (
      if stableAbi then
        lib.versionAtLeast python3.version python.version
        && (python.abi == "abi3t" || !(isFreeThreaded python3))
      else
        lib.versions.majorMinor python3.version == lib.versions.majorMinor python.version
        && isFreeThreaded python3 == (python.abi == "nogil")
    ) message;

  applicableBuildSets = lib.filter checkPython buildSets;

  variantOf = buildSet: buildSet.variants.sdkVariant python;

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
    pkgs.stdenv.mkDerivation {
      name = "${name}-${variant}";

      dontUnpack = true;

      # Keep the binaries as the SDK built them.
      dontFixup = true;

      nativeBuildInputs = [
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

      doAbiCheck = true;
      pythonAbiVersion = if stableAbi then python.version else "none";

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
}
