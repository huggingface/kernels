{
  lib,
  callPackage,
  stdenv,
}:

let
  versions = {
    "3.6.0" = {
      x86_64-linux = {
        url = "https://download.pytorch.org/whl/triton_rocm-3.6.0-cp314-cp314-linux_x86_64.whl";
        hash = "sha256-935tqCKop24JegYeYaf++K4KM+i2mJSYc24W7RIPxvg=";
      };
    };
    "3.7.0" = {
      x86_64-linux = {
        url = "https://download-r2.pytorch.org/whl/triton_rocm-3.7.0-cp314-cp314-linux_x86_64.whl";
        hash = "sha256-DyWfmBMQyEEfPHzX0LvtBagFUMO+dX5tnadDdGsBbmc=";
      };
    };
    "3.7.1" = {
      x86_64-linux = {
        url = "https://download.pytorch.org/whl/triton_rocm-3.7.1-cp314-cp314-linux_x86_64.whl";
        hash = "sha256-kWWwQuNe3RIB9eV5+mB0p4MoCxNyh/R+TNzYUMRjOlQ=";
      };
    };
    "3.8.0" = {
      x86_64-linux = {
        url = "https://download.pytorch.org/whl/triton_rocm-3.8.0-cp314-cp314-linux_x86_64.whl";
        hash = "sha256-uyeJBwRNLXGyZNTrogX76h343PBR03tLNENacElx2ZU=";
      };
    };
    "3.9.0" = {
      x86_64-linux = {
        url = "https://download.pytorch.org/whl/triton_rocm-3.9.0-cp312-abi3-linux_x86_64.whl";
        hash = "sha256-lOTCKOqQpNbHntxQY4Mpk9zrNdgiAtHIkejsjiv1Q8M=";
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
