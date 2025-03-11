{
  inputs = {
    tgi-nix.url = "github:huggingface/text-generation-inference-nix/kernels-0.2.0";
    nixpkgs.follows = "tgi-nix/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      tgi-nix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          inherit (tgi-nix.lib) config;
          overlays = [
            tgi-nix.overlays.default
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
                huggingface-hub
                pytest
                pytest-benchmark
                torch
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
