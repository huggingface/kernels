{
  callPackage,
  fetchurl,
  stdenvNoCC,
}:

let
  generic = callPackage ./generic.nix { };
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
  aotriton_0_11_2 = generic rec {
    version = "0.11.2b";

    hashes = {
      "7.1" = "sha256-/uNr6z6khM4YFVu6/gJsV3/WcF5EaeWUBbJgvXS4zBA=";
      "7.2" = "sha256-zYq/J7u2POxFyUE16bKHRZZgdCY6awVV5YeK4ctqI0k=";
    };

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
  };

  aotriton_0_12 = generic rec {
    version = "0.12b";

    hashes = {
      "7.1" = "sha256-odcxdFkpthWY0IjuqtMdioKicDKqUeOnyDHkWpnglcI=";
      "7.2" = "sha256-W5fo0EGxYMhAhZYfPTvXuYkGQrFGussEyZGqmtao3Kg=";
    };

    images = mkImages version [
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.12b/aotriton-0.12b-images-amd-gfx90a.tar.gz";
        hash = "sha256-u4vyI3t3/FA7wpZ+oNmdbKQZEmxHnpUepCtxJzcSgIY=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.12b/aotriton-0.12b-images-amd-gfx942.tar.gz";
        hash = "sha256-8I7az4PJzPHEvctR8cqwUtFoCr6jHJ4DXz+frbLxO6Q=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.12b/aotriton-0.12b-images-amd-gfx950.tar.gz";
        hash = "sha256-MHo31ynNo6ISBEmQnlGSzXHCutzL038CInhgmOacepE=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.12b/aotriton-0.12b-images-amd-gfx110x.tar.gz";
        hash = "sha256-ycrHz28ncWjhZZrC8EcG+II1gLfH4+iV9aVQPta91V8=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.12b/aotriton-0.12b-images-amd-gfx115x.tar.gz";
        hash = "sha256-MXc4ehXGeLMAV/RYTR/BuPjbVhY4kMtcmPJ0UCCfWns=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.12b/aotriton-0.12b-images-amd-gfx120x.tar.gz";
        hash = "sha256-aFclEc5kh6g/kBS9JVvWnIlD+H0Mk71XstqsX7xsecE=";
      })
    ];
  };

  aotriton_0_13 = generic rec {
    version = "0.13b";

    hashes = {
      "7.2" = "sha256-HN7rt+9hq2kfuh2B2pGbnbXYvvKCaciSowvROgSVt6A=";
      "7.14" = "sha256-ehOXl8FrAC/V2bzXBtNtyYGbsQiHcVD4GG2iHQWQ6qY=";
    };

    images = mkImages version [
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.13b/aotriton-0.13b-images-amd-gfx90a.tar.gz";
        hash = "sha256-o9GmhozikLqBGGGCBwk+eFJS7/ThimT0lXUstaA//tY=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.13b/aotriton-0.13b-images-amd-gfx942.tar.gz";
        hash = "sha256-zNvH49loOb5Ile4ATyFTHMVdWQyQGJN7njFLujY7OSc=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.13b/aotriton-0.13b-images-amd-gfx950.tar.gz";
        hash = "sha256-UY/QcusFlI/ApsJaIIMlkcZAbfhl47aRoq7/P9TFzh0=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.13b/aotriton-0.13b-images-amd-gfx110x.tar.gz";
        hash = "sha256-7+dz56LIrcmV2Q7NDayi24AShURe4QQy3yspxn1bEdI=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.13b/aotriton-0.13b-images-amd-gfx115x.tar.gz";
        hash = "sha256-G8UOiqi2vaNBDpKIbMyo/UXfPmCmy9qf/FiyxUHv1cI=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.13b/aotriton-0.13b-images-amd-gfx120x.tar.gz";
        hash = "sha256-akZdvAMUi7qKLXjEwqPLgxVeygD09/dJ5nZALXZglow=";
      })
      (fetchurl {
        url = "https://github.com/ROCm/aotriton/releases/download/0.13b/aotriton-0.13b-images-amd-gfx1250.tar.gz";
        hash = "sha256-Sq9x1uUQVJ1ZN1fl+IWY3x5KKcvNkfcHUO6KdvZcAn8=";
      })
    ];
  };

}
