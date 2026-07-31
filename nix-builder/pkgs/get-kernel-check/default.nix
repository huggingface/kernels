{
  makeSetupHook,
  proot,
  python3,
}:

makeSetupHook {
  name = "get-kernel-check-hook";
  substitutions = {
    proot = "${proot}/bin/proot";
    python3 = "${python3}/bin/python";
    kernels = "${with python3.pkgs; makePythonPath [ kernels ]}";
  };
} ./get-kernel-check-hook.sh
