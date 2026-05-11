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
  aotriton_0_11_1 = generic rec {
    version = "0.11.1b";

    hashes = {
      "7.0" = "sha256-3rgEbp75dsJzn9BWO1AjnhLcAC19T5fBxKGHSstlq8Q=";
      "7.1" = "sha256-wWE+2enuzHNZ8EoWJLtSjlT15jaeaC3URuqpNtlFI1g=";
      "7.2" = "sha256-VsoxJUwWVfpNUWji2zFZeBwkQtX2sBiCFV6ThZuFzxY=";
    };

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
  };

  aotriton_0_11_2 = generic rec {
    version = "0.11.2b";

    hashes = {
      "7.0" = "sha256-VQGgo7MAiQABtmJfKjU5p7rWDzhvCgYevn1O1coPr7k=";
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

}
