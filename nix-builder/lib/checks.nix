{
  lib,
  runCommand,
  testers,
  python3,

  build,
  buildSets,
}:

let
  kernelBuildSets = build.applicableBuildSets {
    inherit buildSets;
    path = ../../examples/kernels/relu-torch-bounds;
  };

  badRegistrationCheck = testers.testBuildFailure' {
    drv = runCommand "check-bad-registration" { buildInputs = [ python3 ]; } ''
      python3 ${../pkgs/torch-ops-check/torch-ops-check-hook.py} \
        ${../../examples/kernels/silu-and-mul-bad-registration}
      touch $out
    '';
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
