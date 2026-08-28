{
  # Needs to be from the `final` set to get all overrides.
  callPackage,

  # Needs to be from the `prev` set. `lib` functions are used to create
  # the set structure, so using lib from `final` will lead to an infinite
  # recursion.
  lib,
}:

let
  flattenVersion = lib.replaceStrings [ "." ] [ "_" ];
  mkPackages = callPackage ./mk-packages.nix { };
  manifests =
    lib.mapAttrs'
      (
        fileName: type:
        let
          version = lib.removeSuffix ".json" fileName;
        in
        {
          name = version;
          value = lib.importJSON (./manifests/${fileName});
        }
      )

      (builtins.readDir ./manifests);

in
lib.mapAttrs' (version: manifest: {
  name = "xpuPackages_${flattenVersion version}";
  value = mkPackages manifest;
}) manifests
