{ makeSetupHook, python3 }:

makeSetupHook {
  name = "torch-ops-check-hook";
  substitutions = {
    inherit python3;
    hook = ./torch-ops-check-hook.py;
  };
} ./torch-ops-check-hook.sh
