{
  nixpkgs,
  rust-overlay,
}:

let
  inherit (nixpkgs) lib;

  overlay = import ../../nix/overlay.nix;

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

  mkBuildSet = import ./mk-build-set.nix { inherit nixpkgs rust-overlay; };

in
rec {
  mkBuildSets =
    torchVersions: systems:
    lib.concatMap (system: builtins.map mkBuildSet (buildConfigs torchVersions system)) systems;

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
