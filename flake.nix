{
  description = "Kernels";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
    flake-compat.url = "github:edolstra/flake-compat";
  };

  outputs =
    {
      self,
      flake-compat,
      flake-utils,
      nixpkgs,
    }:
    let
      systems = with flake-utils.lib.system; [
        aarch64-darwin
        aarch64-linux
        x86_64-linux
      ];

      torchVersions' = import ./builder/versions.nix;

      # Create an attrset { "<system>" = [ <buildset> ...]; ... }.
      mkBuildSetsPerSystem =
        torchVersions:
        builtins.listToAttrs (
          builtins.map (system: {
            name = system;
            value = import ./builder/lib/build-sets.nix {
              inherit nixpkgs system torchVersions;
            };
          }) systems
        );

      defaultBuildSetsPerSystem = mkBuildSetsPerSystem torchVersions';

      mkBuildPerSystem =
        buildSetPerSystem:
        builtins.mapAttrs (
          system: buildSet: nixpkgs.legacyPackages.${system}.callPackage builder/lib/build.nix { }
        ) buildSetPerSystem;

      defaultBuildPerSystem = mkBuildPerSystem defaultBuildSetsPerSystem;

      # The lib output consists of two parts:
      #
      # - Per-system build functions.
      # - `genFlakeOutputs`, which can be used by downstream flakes to make
      #   standardized outputs (for all supported systems).
      lib = {
        allBuildVariantsJSON =
          let
            buildVariants =
              (import ./builder/lib/build-variants.nix {
                inherit (nixpkgs) lib;
              }).buildVariants
                torchVersions';
          in
          builtins.toJSON buildVariants;
        genFlakeOutputs =
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
            buildSetPerSystem = mkBuildSetsPerSystem (torchVersions torchVersions');
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
      }
      // defaultBuildPerSystem;
    in
    flake-utils.lib.eachSystem systems (
      system:
      let
        # Plain nixkpgs that we use to access utility functions.
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (nixpkgs) lib;

        buildSets = defaultBuildSetsPerSystem.${system};

      in
      rec {
        checks.default = pkgs.callPackage ./builder/lib/checks.nix {
          inherit buildSets;
          build = defaultBuildPerSystem.${system};
        };

        formatter = pkgs.nixfmt-tree;

        packages =
          let
            # Dependencies that should be cached, the structure of the output
            # path is: <build variant>/<dependency>-<output>
            mkForCache =
              buildSets:
              let
                filterDist = lib.filter (output: output != "dist");
                # Get all outputs except for `dist` (which is the built wheel for Torch).
                allOutputs =
                  drv:
                  map (output: {
                    name = "${drv.pname or drv.name}-${output}";
                    path = drv.${output};
                  }) (filterDist drv.outputs or [ "out" ]);
                buildSetOutputs =
                  buildSet:
                  with buildSet.pkgs;
                  (
                    allOutputs buildSet.torch
                    ++ lib.concatMap allOutputs buildSet.extension.extraBuildDeps
                    ++ allOutputs build2cmake
                    ++ allOutputs kernel-abi-check
                    ++ allOutputs python3Packages.kernels
                    ++ lib.optionals stdenv.hostPlatform.isLinux (allOutputs stdenvGlibc_2_27)
                  );
                buildSetLinkFarm = buildSet: pkgs.linkFarm buildSet.torch.variant (buildSetOutputs buildSet);
              in
              pkgs.linkFarm "packages-for-cache" (
                map (buildSet: {
                  name = buildSet.torch.variant;
                  path = buildSetLinkFarm buildSet;
                }) buildSets
              );

          in
          rec {
            build2cmake = pkgs.callPackage ./nix/pkgs/build2cmake { };

            kernel-abi-check = pkgs.callPackage ./nix/pkgs/kernel-abi-check { };

            update-build = pkgs.writeShellScriptBin "update-build" ''
              ${build2cmake}/bin/build2cmake update-build ''${1:-build.toml}
            '';

            forCache = mkForCache (
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
