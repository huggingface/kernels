{
  callPackage,
  fetchFromGitHub,
  fetchpatch,
  fetchurl,
  stdenvNoCC,
}:

let
  generic = callPackage ./generic.nix { };
  postFetch = ''
    cd $out                                                                                                                                                                                                                        
    git reset --hard HEAD                                                                                                                                                                                                          
    for submodule in $(git config --file .gitmodules --get-regexp path | awk '{print $2}' | grep '^third_party/' | grep -v '^third_party/triton$'); do                                                                             
      git submodule update --init --recursive "$submodule"                                                                                                                                                                         
    done                                                                                                                                                                                                                           
    find "$out" -name .git -print0 | xargs -0 rm -rf                                                                                                                                                                               
  '';
  mkImages =
    version: srcs:
    stdenvNoCC.mkDerivation {
      name = "images-${version}";

      inherit srcs;

      buildCommand = ''
        mkdir -p $out
        for src in $srcs; do
          tar -C $out -zxf $src --strip-component=1 --wildcards "aotriton/lib/aotriton.images/*/"
        done
      '';
    };
in
{
  aotriton_0_11_1 = generic rec {
    version = "0.11.1b";

    src = fetchFromGitHub {
      owner = "ROCm";
      repo = "aotriton";
      rev = version;
      hash = "sha256-F7JjyS+6gMdCpOFLldTsNJdVzzVwd6lwW7+V8ZOZfig=";
      leaveDotGit = true;
      inherit postFetch;
    };

    patches = [
      # Fails with: ld.lld: error: unable to insert .comment after .comment
      ./v0.11.1b-no-ld-script.diff
    ];

    gpuTargets = [
      # aotriton GPU support list:
      # https://github.com/ROCm/aotriton/blob/main/v2python/gpu_targets.py
      "gfx90a"
      "gfx942"
      "gfx950"
      "gfx1100"
      "gfx1151"
      "gfx1201"
    ];

    images = mkImages version [
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.11.1b/aotriton-0.11.1b-images-amd-gfx90a.tar.gz";
        hash = "sha256-/p8Etmv1KsJ80CXh2Jz9BJdN0/s64HYZL3g2QaTYD98=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.11.1b/aotriton-0.11.1b-images-amd-gfx942.tar.gz";
        hash = "sha256-CnvO4Z07ttVIcyJIwyNPe5JzbCq3p6rmUpS4en/WTAY=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.11.1b/aotriton-0.11.1b-images-amd-gfx950.tar.gz";
        hash = "sha256-wbo7/oQhf9Z9890fi2fICn97M9CtTXS0HWVnA24DKs4=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.11.1b/aotriton-0.11.1b-images-amd-gfx11xx.tar.gz";
        hash = "sha256-ZjIEDEBdgzvm/3ICkknHdoOLr18Do8E7pOjTeoe3p0A=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.11.1b/aotriton-0.11.1b-images-amd-gfx120x.tar.gz";
        hash = "sha256-Ck/zJL/9rAwv3oeop/cFY9PISoCtTo8xNF8rQKE4TpU=";
      })
    ];

    extraPythonDepends = ps: [ ps.pandas ];
  };

  aotriton_0_11_2 = generic rec {
    version = "0.11.2b";

    src = fetchFromGitHub {
      owner = "ROCm";
      repo = "aotriton";
      rev = version;
      hash = "sha256-VIwwQR1fl40NLNOwO8KhQK/xOK6wb2l8qBugJ1cRjm4=";
      leaveDotGit = true;
      inherit postFetch;
    };

    patches = [
      # Fails with: ld.lld: error: unable to insert .comment after .comment
      ./v0.11.1b-no-ld-script.diff
    ];

    gpuTargets = [
      # aotriton GPU support list:
      # https://github.com/ROCm/aotriton/blob/main/v2python/gpu_targets.py
      "gfx90a"
      "gfx942"
      "gfx950"
      "gfx1100"
      "gfx1151"
      "gfx1201"
    ];

    images = mkImages version [
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.11.2b/aotriton-0.11.2b-images-amd-gfx90a.tar.gz";
        hash = "sha256-/p8Etmv1KsJ80CXh2Jz9BJdN0/s64HYZL3g2QaTYD98=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.11.2b/aotriton-0.11.2b-images-amd-gfx942.tar.gz";
        hash = "sha256-CnvO4Z07ttVIcyJIwyNPe5JzbCq3p6rmUpS4en/WTAY=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.11.2b/aotriton-0.11.2b-images-amd-gfx950.tar.gz";
        hash = "sha256-wbo7/oQhf9Z9890fi2fICn97M9CtTXS0HWVnA24DKs4=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.11.2b/aotriton-0.11.2b-images-amd-gfx11xx.tar.gz";
        hash = "sha256-g5KZY3/MsT++Pngj1X0bLc0OC+14q7y3AF6l9P2CuSg=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.11.2b/aotriton-0.11.2b-images-amd-gfx120x.tar.gz";
        hash = "sha256-Ck/zJL/9rAwv3oeop/cFY9PISoCtTo8xNF8rQKE4TpU=";
      })
    ];

    extraPythonDepends = ps: [ ps.pandas ];
  };

}
