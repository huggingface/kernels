{
  inputs = {
    hf-nix.url = "github:huggingface/hf-nix/torch-cxx11";
    nixpkgs.follows = "hf-nix/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      hf-nix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          inherit (hf-nix.lib) config;
          overlays = [
            hf-nix.overlays.default
          ];
        };
      in
      {
        formatter = pkgs.nixfmt-rfc-style;
        devShells = with pkgs; rec {
          default = mkShell {
            buildInputs =
              [
                black
                mypy
                pyright
                ruff
              ]
              ++ (with python3.pkgs; [
                docutils
                huggingface-hub
                pytest
                pytest-benchmark
                pyyaml
                torch
                types-pyyaml
                venvShellHook
              ]);

            venvDir = "./.venv";

            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
              ( python -m pip install --no-build-isolation --no-dependencies -e . )
            '';
          };
        };
      }
    );
}
