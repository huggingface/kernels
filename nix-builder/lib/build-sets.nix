{
  nixpkgs,
  rust-overlay,

  # Git provenance (`{ sha, dirty }` or `null`) of the `kernel-builder` flake,
  # so it can be burned into the binary.
  builderProvenance ? null,
}:

let
  inherit (nixpkgs) lib;

  inherit (import ./torch-version-utils.nix { inherit lib; })
    backend
    flattenSystems
    ;

  # All build configurations supported by Torch.
  buildConfigs =
    torchVersions: system:
    let
      filterMap = f: xs: builtins.filter (x: x != null) (builtins.map f xs);
      systemBuildConfigs = filterMap (version: if version.system == system then version else null) (
        flattenSystems torchVersions
      );
    in
    builtins.map (buildConfig: buildConfig // { backend = backend buildConfig; }) systemBuildConfigs;

  inherit
    (import ./mk-build-set.nix {
      inherit
        nixpkgs
        rust-overlay
        builderProvenance
        ;
    })
    pkgsKey
    mkPkgs
    mkBuildSet
    ;

in
rec {
  mkBuildSets =
    torchVersions: systems:
    let
      configs = lib.concatMap (buildConfigs torchVersions) systems;
      # Multiple builsets can share the same nixpkgs instance, for example:
      #
      # - All Torch CPU buildsets.
      # - All Torch buildsets using CUDA n.m (e.g. 13.0).
      # - All Torch buildsets using ROCm n.m. (e.g. 7.2).
      #
      # So we instantiate the nixpkgs only once for each such category to avoid
      # evaluation nixpkgs more than necessary. Note that due to lazy eval,
      # we won't install nixpkgs more than once for the same key.
      pkgsByKey = builtins.listToAttrs (
        map (buildConfig: {
          name = pkgsKey buildConfig;
          value = mkPkgs buildConfig;
        }) configs
      );
    in
    map (buildConfig: mkBuildSet pkgsByKey.${pkgsKey buildConfig} buildConfig) configs;

  # Partition into an attrset { <system> = [ <buildset> ...]; ... }.
  partitionBuildSetsBySystem = lib.foldl (
    acc: buildSet:
    let
      system = buildSet.buildConfig.system;
    in
    acc
    // {
      ${system} = (acc.${system} or [ ]) ++ [ buildSet ];
    }
  ) { };

  # Partition into an attrset { <system>.<backend> = [ <buildset> ...]; ... }.
  partitionBuildSetsBySystemBackend = lib.foldl (
    acc: buildSet:
    let
      system = buildSet.buildConfig.system;
      backend = buildSet.buildConfig.backend;
    in
    lib.recursiveUpdate acc {
      ${system}.${backend} = (lib.attrByPath [ system backend ] [ ] acc) ++ [ buildSet ];
    }
  ) { };
}
