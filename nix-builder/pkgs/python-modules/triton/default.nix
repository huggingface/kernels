{
  lib,
  callPackage,
  stdenv,
}:

let
  versions = {
    "3.5.0" = {
      x86_64-linux = {
        url = "https://download.pytorch.org/whl/triton-3.5.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-tvbbiVAabcSkkv8oFGDBsVVjQgvJCTR3CqanuA/VHJU=";
      };
      aarch64-linux = {
        url = "https://download.pytorch.org/whl/triton-3.5.0-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl";
        hash = "sha256-BeFFtRpTVzv/JgQx/0D63OCDitmSjF7hiDtT1ZiE4Zg=";
      };
    };
    "3.6.0" = {
      x86_64-linux = {
        url = "https://download.pytorch.org/whl/triton-3.6.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-AHUDn/J3ZUgAg7GhCZmb8nEQ81QvH5+tlfD5Blo22nk=";
      };
      aarch64-linux = {
        url = "https://download.pytorch.org/whl/triton-3.6.0-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl";
        hash = "sha256-WNV9Z5awAEB2MVQzUm/p1K9CBE1DCv3uHmzUKna9bQk=";
      };
    };
    "3.7.0" = {
      x86_64-linux = {
        url = "https://download.pytorch.org/whl/test/triton-3.7.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-yvYg5j6eBA83/eq+Pe1fsvzVYv78SRZp6s2DFWLuNwk=";
      };
      aarch64-linux = {
        url = "https://download.pytorch.org/whl/test/triton-3.7.0-cp313-cp313-linux_aarch64.whl";
        hash = "sha256-LoFYjFyKWMAkMMrTeOgAopRK81mqm+fEQPTuOplp9jY=";
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
  lib.nameValuePair ("triton_${versionAttr version}") (
    generic ((forSystem systems) // { inherit version; })
  )
) versions
