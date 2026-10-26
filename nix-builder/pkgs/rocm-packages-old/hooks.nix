final: prev: {
  markForRocmRootHook = final.callPackage (
    { makeSetupHook }: makeSetupHook { name = "mark-for-rocm-root-hook"; } ./mark-for-rocm-root-hook.sh
  ) { };

  setupRocmHook = (
    final.callPackage (
      { makeSetupHook }:
      makeSetupHook {
        name = "setup-rocm-hook";

        substitutions.setupRocmHook = placeholder "out";
      } ./setup-rocm-hook.sh
    ) { }
  );
}
