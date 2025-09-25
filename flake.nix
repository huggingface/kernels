{
  inputs = {
    hf-nix.url = "github:huggingface/hf-nix";
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
        packages.kernel-abi-check = pkgs.python3.pkgs.callPackage ./nix/kernel-abi-check.nix {};
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
                (callPackage ./nix/kernel-abi-check.nix {})
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
