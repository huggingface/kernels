{ makeSetupHook, kernel-builder }:

makeSetupHook {
  name = "hash-kernel-hook";
  substitutions = {
    kernel_builder = kernel-builder;
  };
} ./hash-kernel-hook.sh
