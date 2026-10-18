{
  lib,
  callPackage,
  stdenv,
}:

let
  versions = {
    "3.6.0" = {
      x86_64-linux = {
        url = "https://download.pytorch.org/whl/triton-3.6.0-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-ttgAvaZm675+whRuP16yce6H+PrHJAEdx/Jd7pv7X38=";
      };
      aarch64-linux = {
        url = "https://download.pytorch.org/whl/triton-3.6.0-cp314-cp314-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl";
        hash = "sha256-IO0vdw9yF7iplMupnnysN+e+c1o5kTRJkLgKtP6pZMQ=";
      };
    };
    "3.7.0" = {
      x86_64-linux = {
        url = "https://download.pytorch.org/whl/triton-3.7.0-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-lrEJlBySIGSt4Nc4Ni4DqTB57H5IMjL2/bqAO3YCFMw=";
      };
      aarch64-linux = {
        url = "https://download-r2.pytorch.org/whl/triton-3.7.0-cp314-cp314-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl";
        hash = "sha256-Zo7ZiY+ZmzY79dep1ZUWzkH41di/D0p3cNLufdD57HE=";
      };
    };
    "3.7.1" = {
      x86_64-linux = {
        url = "https://download.pytorch.org/whl/triton-3.7.1-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-xWy4EDSdNpkCC1t2lUKbm5l1dNqxufRrK2P6I9eVSJQ=";
      };
      aarch64-linux = {
        url = "https://download.pytorch.org/whl/triton-3.7.1-cp314-cp314-linux_aarch64.whl";
        hash = "sha256-8JH+hQZXlxxxkfeqW8kDdmmtBF2p036rx1GSatu0Vt4=";
      };
    };
    "3.8.0" = {
      x86_64-linux = {
        url = "https://download.pytorch.org/whl/triton-3.8.0-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-SJ7+VmQeG8XT3zyfO6m8x7uN7kBpGuiMz1STiHN5UaI=";
      };
      aarch64-linux = {
        url = "https://download.pytorch.org/whl/triton-3.8.0-cp314-cp314-linux_aarch64.whl";
        hash = "sha256-9NoIqNpk/evM/u16gz5HPPt8Zw9YWi/h+vjUm+qz4b0=";
      };
    };
    "3.9.0" = {
      x86_64-linux = {
        url = "https://huggingface.co/buckets/kernels-community/pytorch-rc/resolve/2.15.0/rc3/triton-3.9.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-MZtpln1qLD8tdGH+hsZMJNSoNfok33gQLoAm8PBRtWk=";
      };
      aarch64-linux = {
        url = "https://huggingface.co/buckets/kernels-community/pytorch-rc/resolve/2.15.0/rc3/triton-3.9.0-cp312-abi3-linux_aarch64.whl";
        hash = "sha256-Vax66MKJ6ZHXKl+ylIGHpPsy/Bu4JurO8TIROe7J53I=";
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
