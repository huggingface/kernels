{
  self,
  lib,
  runCommand,
  testers,
  python3,
  stdenv,

  build,
  buildSets,
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
in
assert lib.assertMsg (builtins.all (buildSet: buildSet.torch.version == "2.10.0") kernelBuildSets)
  ''
    Torch minver/maxver filtering does not work.
  '';
runCommand "builder-nix-checks" { buildInputs = [ badRegistrationCheck ]; } ''
  touch $out
''
