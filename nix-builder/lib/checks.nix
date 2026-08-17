{
  self,
  lib,
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
assert lib.assertMsg (builtins.all (buildSet: buildSet.torch.version == "2.12.0") kernelBuildSets)
  ''
    Torch minver/maxver filtering does not work.
  '';
runCommand "builder-nix-checks"
  {
    buildInputs = [
      badRegistrationCheck
      fetchFromHuggingFaceCheck
    ];
  }
  ''
    touch $out
  ''
