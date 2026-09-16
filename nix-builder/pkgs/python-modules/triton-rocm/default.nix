{
  lib,
  callPackage,
  stdenv,
}:

let
  versions = {
    "3.7.0" = {
      x86_64-linux = {
        url = "https://download-r2.pytorch.org/whl/triton_rocm-3.7.0-cp314-cp314-linux_x86_64.whl";
        hash = "sha256-DyWfmBMQyEEfPHzX0LvtBagFUMO+dX5tnadDdGsBbmc=";
      };
    };
  };
  generic = callPackage ./generic.nix { };
  versionAttr = lib.replaceStrings [ "." ] [ "_" ];
  forSystem =
    systems:
    systems.${stdenv.hostPlatform.system}
      or (builtins.throw "System `${stdenv.hostPlatform.system}` is not supported by the triton package");
in
lib.mapAttrs' (
  version: systems:
  lib.nameValuePair ("triton-rocm_${versionAttr version}") (
    generic ((forSystem systems) // { inherit version; })
  )
) versions
