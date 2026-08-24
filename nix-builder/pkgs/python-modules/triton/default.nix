{
  lib,
  callPackage,
  stdenv,
}:

let
  versions = {
    "3.7.0" = {
      x86_64-linux = {
        url = "https://download.pytorch.org/whl/triton-3.7.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-0BA4KB83Yz8GFC8dTazEh+pNRRsC6zPt/edY0TZii6I=";
      };
      aarch64-linux = {
        url = "https://download-r2.pytorch.org/whl/triton-3.7.0-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl";
        hash = "sha256-m4KV9sctz4QNP0ysJcGy/0gu8enq0jQVtI/C2Dc6Fw0=";
      };
    };
    "3.7.1" = {
      x86_64-linux = {
        url = "https://download.pytorch.org/whl/triton-3.7.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-WWiLm5JPiHMW3ND66ejL5pfuHR9qs4ZyOYLMns3r7gE=";
      };
      aarch64-linux = {
        url = "https://download.pytorch.org/whl/triton-3.7.1-cp313-cp313-linux_aarch64.whl";
        hash = "sha256-NIlNUa/xq/ewF/vQxen+chEwXDWZFX/KJR4/QryPAM8=";
      };
    };
    "3.8.0" = {
      x86_64-linux = {
        url = "https://huggingface.co/buckets/danieldk/pytorch-rc/resolve/2.14.0/rc6/triton-3.8.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-N5GzFsE7IramrxGuYTCvFteG/LORzv0vS38JSqfnQH8=";
      };
      aarch64-linux = {
        url = "https://huggingface.co/buckets/danieldk/pytorch-rc/resolve/2.14.0/rc6/triton-3.8.0-cp313-cp313-linux_aarch64.whl";
        hash = "sha256-wSLjP1kxqfERDKia4OpaeXkkGw3RVXDsSfKv9qyLcd0=";
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
