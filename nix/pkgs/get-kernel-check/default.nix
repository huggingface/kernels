{ makeSetupHook, python3 }:

makeSetupHook {
  name = "get-kernel-check-hook";
  substitutions = {
    python3 = "${python3}/bin/python";
    kernels = "${with python3.pkgs; makePythonPath [ kernels ]}";
  };
} ./get-kernel-check-hook.sh
