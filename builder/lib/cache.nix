{
  # Dependencies that should be cached, the structure of the output
  # path is: <build variant>/<dependency>-<output>
  mkForCache =
    pkgs: buildSets:
    let
      inherit (pkgs) lib;
      filterDist = lib.filter (output: output != "dist");
      # Get all outputs except for `dist` (which is the built wheel for Torch).
      allOutputs =
        drv:
        map (output: {
          name = "${drv.pname or drv.name}-${output}";
          path = drv.${output};
        }) (filterDist drv.outputs or [ "out" ]);
      buildSetOutputs =
        buildSet:
        with buildSet.pkgs;
        (
          allOutputs buildSet.torch
          ++ lib.concatMap allOutputs buildSet.extension.extraBuildDeps
          ++ allOutputs build2cmake
          ++ allOutputs kernel-abi-check
          ++ allOutputs python3Packages.kernels
          ++ lib.optionals stdenv.hostPlatform.isLinux (allOutputs stdenvGlibc_2_27)
        );
      buildSetLinkFarm = buildSet: pkgs.linkFarm buildSet.torch.variant (buildSetOutputs buildSet);
    in
    pkgs.linkFarm "packages-for-cache" (
      map (buildSet: {
        name = buildSet.torch.variant;
        path = buildSetLinkFarm buildSet;
      }) buildSets
    );

}
