final: prev: {
  markForXpuRootHook = final.callPackage (
    { makeSetupHook }: makeSetupHook { name = "mark-for-xpu-root-hook"; } ./mark-for-xpu-root-hook.sh
  ) { };

  setupXpuHook = (
    final.callPackage (
      { makeSetupHook }:
      makeSetupHook {
        name = "setup-xpu-hook";

        substitutions.setupXpuHook = placeholder "out";
      } ./setup-xpu-hook.sh
    ) { }
  );
}
