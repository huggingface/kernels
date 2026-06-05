{ makeSetupHook, kernel-builder }:

makeSetupHook {
  name = "remove-bytecode-hook";
  substitutions = {
    kernel_builder = kernel-builder;
  };
} ./hash-kernel-hook.sh
