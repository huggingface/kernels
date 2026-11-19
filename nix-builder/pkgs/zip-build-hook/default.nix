{ makeSetupHook, kernel-builder }:

makeSetupHook {
  name = "zip-build-hook";
  substitutions = {
    kernel_builder = kernel-builder;
  };
} ./zip-build-hook.sh
