{
  config,
  lib,
  makeSetupHook,
  stdenv,

  proot,
  python3,

  kernels ? python3.pkgs.kernels,
}:

let
  useFakeSys = config.tpuSupport or false;
in
makeSetupHook {
  name = "get-kernel-check-hook";
  substitutions = {
    python3 = "${python3}/bin/python";
    kernels = "${python3.pkgs.makePythonPath [ kernels ]}";
    proot = lib.optionalString useFakeSys "${proot}/bin/proot";
    pyhook = ./get-kernel-check-hook.py;
    useFakeSys = lib.optionalString useFakeSys "1";
  };
} ./get-kernel-check-hook.sh
