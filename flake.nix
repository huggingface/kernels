{
  inputs = {
    hf-nix.url = "github:huggingface/hf-nix/mktestdocs-0.2.5";
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
          config = hf-nix.lib.config system;
          overlays = [
            hf-nix.overlays.default
          ];
        };
      in
      {
        formatter = pkgs.nixfmt-tree;
        devShells = with pkgs; rec {
          default = mkShell {
            nativeBuildInputs = [
              # For hf-doc-builder.
              nodejs
            ];
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
                mktestdocs
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
