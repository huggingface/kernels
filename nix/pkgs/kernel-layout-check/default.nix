{ makeSetupHook, python3 }:

makeSetupHook {
  name = "kernel-layout-check-hook";
} ./kernel-layout-check-hook.sh
