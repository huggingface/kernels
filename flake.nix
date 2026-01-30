{
  description = "Kernels";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
    flake-compat.url = "github:edolstra/flake-compat";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      flake-compat,
      flake-utils,
      nixpkgs,
      rust-overlay,
    }:
    let
      inherit
        (import ./builder/lib/build-sets.nix {
          inherit nixpkgs rust-overlay;
        })
        mkBuildSets
        partitionBuildSetsBySystem
        partitionBuildSetsBySystemBackend
        ;
      inherit (import ./builder/lib/cache.nix) mkForCache;

      systems = with flake-utils.lib.system; [
        aarch64-darwin
        aarch64-linux
        x86_64-linux
      ];

      torchVersions' = import ./builder/versions.nix;

      defaultBuildSets = mkBuildSets torchVersions' systems;
      defaultBuildSetsPerSystem = partitionBuildSetsBySystem defaultBuildSets;

      mkBuildPerSystem =
        systems:
        builtins.listToAttrs (
          builtins.map (system: {
            name = system;
            value = nixpkgs.legacyPackages.${system}.callPackage builder/lib/build.nix { };
          }) systems
        );
      buildPerSystem = mkBuildPerSystem systems;

      # The lib output consists of two parts:
      #
      # - Per-system build functions.
      # - `genFlakeOutputs`, which can be used by downstream flakes to make
      #   standardized outputs (for all supported systems).
      lib = rec {
        allBuildVariantsJSON =
          let
            buildVariants =
              (import ./builder/lib/build-variants.nix {
                inherit (nixpkgs) lib;
              }).buildVariants
                torchVersions';
          in
          builtins.toJSON buildVariants;
        genFlakeOutputs = builtins.warn ''
          `genFlakeOutputs` was renamed to `genKernelFlakeOutputs` and will be removed
          in kernel-builder 0.14.
        '' genKernelFlakeOutputs;
        genKernelFlakeOutputs =
          {
            path,
            rev ? null,
            self ? null,

            # This option is not documented on purpose. You should not use it,
            # if a kernel cannot be imported, it is non-compliant. This is for
            # one exceptional case: packaging a third-party kernel (where you
            # want to stay close to upstream) where importing the kernel will
            # fail in a GPU-less sandbox. Even in that case, it's better to lazily
            # load the part with this functionality.
            doGetKernelCheck ? true,
            pythonCheckInputs ? pkgs: [ ],
            pythonNativeCheckInputs ? pkgs: [ ],
            torchVersions ? _: torchVersions',
          }:
          assert
            (builtins.isFunction torchVersions)
            || abort "`torchVersions` must be a function taking one argument (the default version set)";
          let
            buildSets = mkBuildSets (torchVersions torchVersions') systems;
            buildSetPerSystem = partitionBuildSetsBySystem buildSets;
            buildPerSystem = mkBuildPerSystem buildSetPerSystem;
          in
          flake-utils.lib.eachSystem systems (
            system:
            nixpkgs.legacyPackages.${system}.callPackage ./builder/lib/gen-flake-outputs.nix {
              inherit
                system
                path
                rev
                self
                doGetKernelCheck
                pythonCheckInputs
                pythonNativeCheckInputs
                ;
              build = buildPerSystem.${system};
              buildSets = buildSetPerSystem.${system};
            }
          );
      };
      #// defaultBuildPerSystem;
    in
    flake-utils.lib.eachSystem systems (
      system:
      let
        # Plain nixkpgs that we use to access utility functions.
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (nixpkgs) lib;

        buildSets = defaultBuildSetsPerSystem.${system};
        buildSetsByBackend = (partitionBuildSetsBySystemBackend defaultBuildSets).${system};
        buildSet = builtins.head buildSetsByBackend.cuda;

        # Dev shells per framework.
        devShellByBackend = lib.mapAttrs (
          backend: buildSet:
          with (builtins.head buildSet).pkgs;
          let
            rust = rust-bin.stable.latest.default.override {
              extensions = [
                "rust-analyzer"
                "rust-src"
              ];
            };
          in
          mkShell {
            nativeBuildInputs = [
              build2cmake
              kernel-abi-check
              nodejs # For hf-doc-builder.
              pkg-config
              rust
            ];
            buildInputs = [
              black
              mypy
              pyright
              ruff
            ]
            ++ (with python3.pkgs; [
              docutils
              huggingface-hub
              kernel-abi-check
              mktestdocs
              openssl.dev
              pytest
              pytest-benchmark
              pyyaml
              torch
              types-pyyaml
              venvShellHook
            ]);

            RUST_SRC_PATH = "${rust}/lib/rustlib/src/rust/library";

            venvDir = "./.venv";

            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
              ( python -m pip install --no-build-isolation --no-dependencies -e kernels )
            '';

          }
        ) buildSetsByBackend;
      in
      rec {
        checks.default = pkgs.callPackage ./builder/lib/checks.nix {
          inherit buildSets;
          build = buildPerSystem.${system};
        };

        devShells = devShellByBackend // {
          default = devShellByBackend.${if system == "aarch64-darwin" then "metal" else "cuda"};
        };

        formatter = pkgs.nixfmt-tree;

        packages = rec {
          inherit (buildSet.pkgs) build2cmake kernel-abi-check;
          inherit (buildSet.pkgs.python3.pkgs) kernels;

          update-build = pkgs.writeShellScriptBin "update-build" ''
            ${build2cmake}/bin/build2cmake update-build ''${1:-build.toml}
          '';

          forCache = mkForCache pkgs (
            builtins.filter (buildSet: buildSet.buildConfig.bundleBuild or false) buildSets
          );

          forCacheNonBundle = mkForCache (
            builtins.filter (buildSet: !(buildSet.buildConfig.bundleBuild or false)) buildSets
          );

          # This package set is exposed so that we can prebuild the Torch versions.
          torch = builtins.listToAttrs (
            map (buildSet: {
              name = buildSet.torch.variant;
              value = buildSet.torch;
            }) buildSets
          );
        };
      }
    )
    // {
      inherit lib;
    };
}
