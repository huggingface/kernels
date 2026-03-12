{ makeSetupHook, python3 }:

makeSetupHook {
  name = "remove-bytecode-hook";
} ./remove-bytecode-hook.sh
