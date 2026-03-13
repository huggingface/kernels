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
  aotriton_0_10 = generic rec {
    version = "0.10b";

    src = fetchFromGitHub {
      owner = "ROCm";
      repo = "aotriton";
      rev = version;
      hash = "sha256-stAHnsqChkNv69wjlhM/qUetrJpNwI1i7rGnPMwsNz0=";
      leaveDotGit = true;
      inherit postFetch;
    };

    patches = [
      # A bunch of implicit type narrowing issues that are rejected by newer
      # compilers.
      ./v0.10b-explicit-cast-for-narrowing.diff
      # Fails with: ld.lld: error: unable to insert .comment after .comment
      ./v0.10b-no-ld-script.diff

      # CMakeLists.txt: AOTRITON_INHERIT_SYSTEM_SITE_TRITON flag
      (fetchpatch {
        url = "https://github.com/ROCm/aotriton/commit/9734c3e999c412a07d2b35671998650942b26ed4.patch";
        hash = "sha256-tBmjjhRJmLv3K6F2+4OcMuwf8dH7efPPECMQjh6QdUA=";
      })
    ];

    gpuTargets = [
      # aotriton GPU support list:
      # https://github.com/ROCm/aotriton/blob/main/v2python/gpu_targets.py
      "gfx90a"
      "gfx942"
      "gfx950"
      "gfx1100"
      "gfx1101"
      "gfx1201"
    ];

    images = mkImages version [
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.10b/aotriton-0.10b-manylinux_2_28_x86_64-rocm6.3-shared.tar.gz";
        hash = "sha256-hhzZ90ee7JQ5M8J8uGkgJH5bXdE5vHwTdsgYCKu31/4=";
      })
    ];

    extraPythonDepends = ps: [ ps.pandas ];
  };

  aotriton_0_11 = generic rec {
    version = "0.11b";

    src = fetchFromGitHub {
      owner = "ROCm";
      repo = "aotriton";
      rev = version;
      hash = "sha256-QXkNB3vNmPg4/m23OMuBBX4cjZQ3zQPotaeimFMbclc=";
      leaveDotGit = true;
      inherit postFetch;
    };

    patches = [
      # Fails with: ld.lld: error: unable to insert .comment after .comment
      ./v0.11b-no-ld-script.diff
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
        url = "https://github.com/ROCm/aotriton/releases/download/0.11b/aotriton-0.11b-images-amd-gfx90a.tar.gz";
        hash = "sha256-wZpByUgFEKsy5vsF5u0KODLWsHY08FC4NrdgIAvvpzU=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.11b/aotriton-0.11b-images-amd-gfx942.tar.gz";
        hash = "sha256-OgapmXHd23cDowN48cXWtBRo2SbqUYIRVtG2hXuYW8Q=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.11b/aotriton-0.11b-images-amd-gfx950.tar.gz";
        hash = "sha256-J/wh9nYdV5h6cAQ23ozynL3Z7u6RMY3+1ZbusUfSGa0=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.11b/aotriton-0.11b-images-amd-gfx11xx.tar.gz";
        hash = "sha256-7BNAMghzRBdmlVBdtlk4c3TRkWrf7hbw20fe442chgM=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.11b/aotriton-0.11b-images-amd-gfx120x.tar.gz";
        hash = "sha256-/sBSBXR/9RZJseFRVFJn1aogN7qdAzjK0oaIKRW5QbA=";
      })
    ];

    extraPythonDepends = ps: [ ps.pandas ];
  };

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

}
