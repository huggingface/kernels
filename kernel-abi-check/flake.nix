{
  description = "kernel-abi-check devenv";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      flake-utils,
      nixpkgs,
      rust-overlay,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };
        rust = pkgs.rust-bin.stable.latest.default.override {
          extensions = [
            "rust-analyzer"
            "rust-src"
          ];
        };
      in
      {

        devShells.default =
          with pkgs;
          mkShell {
            buildInputs = [
              black
              openssl.dev
              pkg-config
              ruff
              rust
            ]
            ++ (with python3.pkgs; [
              pytest
              venvShellHook
            ]);

            nativeBuildInputs = with pkgs; [ maturin ];

            RUST_SRC_PATH = "${rust}/lib/rustlib/src/rust/library";
            venvDir = "./.venv";
          };
      }
    );
}
