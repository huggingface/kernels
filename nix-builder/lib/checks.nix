{
  self,
  lib,
  cmake,
  runCommand,
  testers,
  python3,
  stdenv,

  build,
  buildSets,
  fetchFromHuggingFace,
  genKernelFlakeOutputs,
}:

let
  kernelBuildSets = build.applicableBuildSets {
    inherit buildSets;
    path = ../../examples/kernels/relu-torch-bounds;
  };

  # The Torch version that `relu-torch-bounds` is bounded to (minver/maxver).
  expectedTorchVersion = "2.13.0";
  kernelBuildSetTorchVersions = lib.unique (map (buildSet: buildSet.torch.version) kernelBuildSets);

  badRegistrationCheck = testers.testBuildFailure' {
    drv =
      (genKernelFlakeOutputs {
        inherit self;
        path = ../../examples/kernels/silu-and-mul-bad-registration;
      }).packages.${stdenv.hostPlatform.system}.redistributable.torch-cuda;
    expectedBuilderExitCode = 1;
    expectedBuilderLogEntries = [
      "Found Torch library registrations that do not use `add_op_namespace_prefix`:"
    ];
  };

  # Generate a kernel source with `kernel-port`. The fixture upstream is fetched
  # with the hash that `kernel-port` recorded, so Nix verifies that hash.
  portedSrcFixture = ../tests/ported-src;
  mkPortedSrc = (builtins.head buildSets).pkgs.callPackage ./ported-src.nix {
    fetchgit =
      { hash, ... }:
      runCommand "upstream" {
        outputHashMode = "recursive";
        outputHash = hash;
      } "cp -r ${portedSrcFixture + "/upstream"} $out";
  };
  portedSrc = mkPortedSrc portedSrcFixture;

  portedSrcCheck = runCommand "ported-src-check" { } ''
    test -f ${portedSrc.generated}/torch-ext/pkg/py.typed
    test -f ${portedSrc.verified}/torch-ext/pkg/__init__.py
    # `source-set.nix` only keeps `pyext` files, generated sources should as well.
    test ! -e ${portedSrc.verified}/torch-ext/pkg/py.typed
    touch $out
  '';

  # The hash of the generated tree is how a change in `kernel-port` output shows.
  portedSrcStaleProvenanceCheck = testers.testBuildFailure' {
    drv = (mkPortedSrc (portedSrcFixture + "/stale-provenance")).verified;
    expectedBuilderExitCode = 1;
    expectedBuilderLogEntries = [
      "Committed build.toml or port-provenance.json differs"
    ];
  };

  # Test the CUDA/ROCm arch intersection functions used by the CMake
  # templates. `utils.cmake` is passed explicitly since the test script is
  # copied to the store without its sibling files.
  archIntersectionCheck =
    runCommand "arch-intersection-check"
      {
        nativeBuildInputs = [ cmake ];
      }
      ''
        cmake \
          -DUTILS_CMAKE=${../../kernel-builder/src/pyproject/templates/utils.cmake} \
          -P ${../../kernel-builder/tests/arch-intersection.cmake}
        touch $out
      '';

  # These are tests to verify that the validation function for entries
  # from `versions.nix` works correctly.
  torchVersionUtilsTestFailures =
    let
      inherit (import ./torch-version-utils.nix { inherit lib; })
        flattenSystems
        validateFlatTorchVersion
        validateTorchVersion
        validateUniqueTvmFfiVariants
        ;
      throws = expr: !(builtins.tryEval (builtins.deepSeq expr expr)).success;
      validVersion = {
        torchVersion = "2.13";
        cudaVersion = "13.0";
        systems = [
          "x86_64-linux"
          "aarch64-linux"
        ];
        bundleBuild = true;
      };
      flatVersion = {
        torchVersion = "2.13";
        cudaVersion = "13.0";
        system = "x86_64-linux";
        bundleBuild = true;
      };
    in
    lib.runTests {
      testValidateValid = {
        expr = validateTorchVersion validVersion;
        expected = validVersion;
      };
      testValidateMissingTorchVersion = {
        expr = throws (validateTorchVersion (builtins.removeAttrs validVersion [ "torchVersion" ]));
        expected = true;
      };
      testValidateMissingSystems = {
        expr = throws (validateTorchVersion (builtins.removeAttrs validVersion [ "systems" ]));
        expected = true;
      };
      testValidateUnknownAttr = {
        expr = throws (validateTorchVersion (validVersion // { cudaVerison = "13.0"; }));
        expected = true;
      };
      testValidateFlatValid = {
        expr = validateFlatTorchVersion flatVersion;
        expected = flatVersion;
      };
      testValidateFlatRejectsSystems = {
        expr = throws (validateFlatTorchVersion validVersion);
        expected = true;
      };
      testValidateFlatMissingSystem = {
        expr = throws (validateFlatTorchVersion (builtins.removeAttrs flatVersion [ "system" ]));
        expected = true;
      };
      testValidateRejectsSystem = {
        expr = throws (validateTorchVersion flatVersion);
        expected = true;
      };
      testFlattenSystems = {
        expr = flattenSystems [ validVersion ];
        expected = [
          {
            torchVersion = "2.13";
            cudaVersion = "13.0";
            system = "x86_64-linux";
            bundleBuild = true;
          }
          {
            torchVersion = "2.13";
            cudaVersion = "13.0";
            system = "aarch64-linux";
            bundleBuild = true;
          }
        ];
      };
      testFlattenSystemsValidates = {
        expr = throws (flattenSystems [ (validVersion // { cudaVerison = "13.0"; }) ]);
        expected = true;
      };
      testUniqueTvmFfiVariantsValid = {
        expr = validateUniqueTvmFfiVariants [
          (flatVersion // { tvmFfiVersion = "0.1"; })
          (flatVersion // { torchVersion = "2.12"; })
        ];
        expected = [
          (flatVersion // { tvmFfiVersion = "0.1"; })
          (flatVersion // { torchVersion = "2.12"; })
        ];
      };
      testUniqueTvmFfiVariantsRejectsDuplicate = {
        expr = throws (validateUniqueTvmFfiVariants [
          (flatVersion // { tvmFfiVersion = "0.1"; })
          (
            flatVersion
            // {
              torchVersion = "2.12";
              tvmFfiVersion = "0.1";
            }
          )
        ]);
        expected = true;
      };
      # While at it, let's also validate `versions.nix`.
      testDefaultVersionsValid = {
        expr = throws (validateUniqueTvmFfiVariants (flattenSystems (import ../versions.nix)));
        expected = false;
      };
    };

  fetchFromHuggingFaceCheck =
    runCommand "fetch-from-huggingface-check"
      {
        relu = fetchFromHuggingFace {
          repoId = "kernels-community/relu";
          type = "kernel";
          revision = "d649efb56fb249ac8f7a57fa1866728ad0c60e52";
          hash = "sha256-1CM0MGEOCqqnYV989jcUANEbiB727JftjXTdQVUHPwk=";
        };
      }
      ''
        test -d "$relu/build"
        # Download metadata must not leak into the output, it would make the
        # hash unstable.
        test ! -e "$relu/.cache"
        touch $out
      '';
in
assert lib.assertMsg (kernelBuildSets != [ ]) ''
  Torch minver/maxver filtering does not work: expected build sets with Torch ${expectedTorchVersion}, got no build sets.
'';
assert lib.assertMsg
  (builtins.all (buildSet: buildSet.torch.version == expectedTorchVersion) kernelBuildSets)
  ''
    Torch minver/maxver filtering does not work: expected only build sets with Torch ${expectedTorchVersion}, got: ${lib.concatStringsSep ", " kernelBuildSetTorchVersions}
  '';
assert lib.assertMsg (torchVersionUtilsTestFailures == [ ]) ''
  torch-version-utils tests failed: ${builtins.toJSON torchVersionUtilsTestFailures}
'';
runCommand "builder-nix-checks"
  {
    buildInputs = [
      archIntersectionCheck
      badRegistrationCheck
      fetchFromHuggingFaceCheck
      portedSrcCheck
      portedSrcStaleProvenanceCheck
    ];
  }
  ''
    touch $out
  ''
