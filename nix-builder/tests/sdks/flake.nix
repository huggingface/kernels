{
  description = "SDKs that test the kernel builder";

  inputs = {
    kernel-builder.url = "path:../../..";
  };

  outputs =
    { self, kernel-builder }:
    let
      inherit (kernel-builder.inputs) flake-utils nixpkgs;
      inherit (nixpkgs) lib;

      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      # The SDKs are flakes by themselves, but we use their outputs directly with
      # our own inputs.
      sdkOutputs = path: (import path).outputs { inherit self kernel-builder; };
      sdk-noarch = sdkOutputs ./sdk-noarch/flake.nix;
      sdk-noarch-backend = sdkOutputs ./sdk-noarch-backend/flake.nix;
      sdk-arch = sdkOutputs ./sdk-arch/flake.nix;
      sdk-arch-backend = sdkOutputs ./sdk-arch-backend/flake.nix;

      sdkVariants = outputs: system: lib.sort lib.lessThan outputs.packages.${system}.variants;
    in
    flake-utils.lib.eachSystem systems (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        noarchVariants = sdkVariants sdk-noarch system;
        noarchBackendVariants = sdkVariants sdk-noarch-backend system;
        archVariants = sdkVariants sdk-arch system;
        archBackendVariants = sdkVariants sdk-arch-backend system;

        # The SDKs are trivial, so they are tested directly during the build.
        testPython = pkgs.python3.withPackages (ps: [
          ps.pytest
          kernel-builder.packages.${system}.kernels
        ]);
        mkTest =
          {
            name,
            repoId,
            drv,
            tests,
          }:
          pkgs.runCommand "${name}-pytest" { nativeBuildInputs = [ testPython ]; } ''
            export HOME=$TMPDIR
            export LOCAL_KERNELS="${repoId}=${drv}"
            python -m pytest ${tests} -m kernels_ci -p no:cacheprovider
            touch $out
          '';
      in
      assert lib.assertMsg (noarchVariants == [ "sdk" ])
        "A noarch SDK without a backend should only have the `sdk` variant, got: ${toString noarchVariants}";
      assert lib.assertMsg (
        noarchBackendVariants != [ ]
        && lib.all (
          variant: lib.hasPrefix "sdk-" variant && !(lib.hasInfix "cpython" variant)
        ) noarchBackendVariants
      ) "Noarch SDK variants should be `sdk-<backend>`, got: ${toString noarchBackendVariants}";
      assert lib.assertMsg (archVariants == [ "sdk-cpython-3.9-abi3-${system}" ])
        "An arch SDK without a backend should only have the `sdk-cpython-<version>-<abi>-<system>` variant, got: ${toString archVariants}";
      assert lib.assertMsg (archBackendVariants == [ "sdk-cuda-12.6-cpython-3.9-abi3-${system}" ])
        "An arch SDK with a backend should only have the `sdk-<backend>-<version>-cpython-<version>-<abi>-<system>` variants, got: ${toString archBackendVariants}";
      {
        packages.ci-build = pkgs.linkFarm "ci-test-sdks" {
          sdk-noarch = mkTest {
            name = "sdk-noarch";
            repoId = "kernels-test/sdk-noarch";
            drv = sdk-noarch.packages.${system}.redistributable.sdk;
            tests = ./sdk-noarch/tests;
          };
          sdk-noarch-backend = mkTest {
            name = "sdk-noarch-backend";
            repoId = "kernels-test/sdk-noarch-backend";
            drv = sdk-noarch-backend.packages.${system}.redistributable.sdk-cuda;
            tests = ./sdk-noarch-backend/tests;
          };
          sdk-arch-backend = mkTest {
            name = "sdk-arch-backend";
            repoId = "kernels-test/sdk-arch-backend";
            drv =
              sdk-arch-backend.packages.${system}.redistributable."sdk-cuda-12.6-cpython-3.9-abi3-${system}";
            tests = ./sdk-arch-backend/tests;
          };
          sdk-arch = mkTest {
            name = "sdk-arch";
            repoId = "kernels-test/sdk-arch";
            drv = sdk-arch.packages.${system}.redistributable."sdk-cpython-3.9-abi3-${system}";
            tests = ./sdk-arch/tests;
          };
        };
      }
    );
}
