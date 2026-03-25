{
  lib,
  runCommand,

  build,
  buildSets,
}:

let
  kernelBuildSets = build.applicableBuildSets {
    inherit buildSets;
    path = ../../examples/kernels/relu-torch-bounds;
  };
in
assert lib.assertMsg (builtins.all (buildSet: buildSet.torch.version == "2.10.0") kernelBuildSets) ''
  Torch minver/maxver filtering does not work.
'';
runCommand "builder-nix-checks" { } ''
  touch $out
''
