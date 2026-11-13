{
  lib,
  build,

  writeScriptBin,
  runCommand,

  path,
  buildSets,
  rev ? null,
  self ? null,

  doGetKernelCheck,
  pythonCheckInputs,
  pythonNativeCheckInputs,
}:

let
  supportedFormat = ''
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
  '';
  flakeRev =
    if self != null then
      self.shortRev or self.dirtyShortRev or (builtins.throw ''
        Kernel is not in a git repository, this will create a non-reproducible build.
        This will not be supported in the future.
      '' self.lastModifiedDate)
    else if rev != null then
      builtins.throw "`rev` argument of `genKernelFlakeOutputs` is deprecated, pass `self` as follows:\n\n${supportedFormat}" rev
    else
      throw "Flake's `self` must be passed to `genKernelFlakeOutputs` as follows:\n\n${supportedFormat}";

  revUnderscored = builtins.replaceStrings [ "-" ] [ "_" ] flakeRev;

  # Git provenance (`{ sha; dirty; }`, or `null`) of the kernel source, recorded
  # in the build metadata. The Nix sandbox has no `.git`, so it is derived from
  # the kernel flake's `self` here and passed in explicitly. (`kernel-builder`'s
  # own provenance is burned into its binary.)
  kernelProvenance = import ./flake-provenance.nix { inherit lib; } self;

  applicableBuildSets = build.applicableBuildSets { inherit path buildSets; };

  kernelConfig = (import ../kernel-config.nix { inherit lib; }) path;

  variantOf = buildSet: buildSet.variants.kernelVariant kernelConfig;
  archVariantOf = buildSet: buildSet.variants.kernelArchVariant kernelConfig;

  hasRustKernels = lib.any (kernel: kernelConfig.kernelLanguage kernel == "rust") (
    lib.attrValues (kernelConfig.toml.kernel or { })
  );
in
import ./gen-flake-outputs.nix {
  inherit
    lib
    writeScriptBin
    applicableBuildSets
    buildSets
    ;

  repoId = lib.attrByPath [
    "general"
    "hub"
    "repo-id"
  ] (throw "[general.hub] section in build.toml does not have a `repo-id` field.") kernelConfig.toml;
  branch = lib.attrByPath [ "general" "hub" "branch" ] null kernelConfig.toml;

  inherit variantOf archVariantOf;

  mkBundle =
    buildSets:
    build.mkExtensionBundle {
      inherit
        path
        doGetKernelCheck
        kernelProvenance
        buildSets
        ;
      rev = revUnderscored;
    };

  mkRedistributable =
    buildSets:
    build.mkDistTorchExtensions {
      inherit
        path
        doGetKernelCheck
        kernelProvenance
        buildSets
        ;
      bundleOnly = false;
      rev = revUnderscored;
    };

  mkDevShells =
    buildSets:
    build.mkDevShells {
      inherit
        path
        doGetKernelCheck
        pythonCheckInputs
        pythonNativeCheckInputs
        buildSets
        ;
      rev = revUnderscored;
    };

  mkTestShells =
    buildSets:
    build.torchExtensionShells {
      inherit
        path
        doGetKernelCheck
        pythonCheckInputs
        pythonNativeCheckInputs
        buildSets
        ;
      rev = revUnderscored;
    };

  mkCiTests =
    buildSets:
    build.mkCiTests {
      inherit
        path
        doGetKernelCheck
        pythonCheckInputs
        kernelProvenance
        buildSets
        ;
      rev = revUnderscored;
    };

  extraPackages =
    { pkgs, overrideTorch }:
    {
      lock-kernel-deps = writeScriptBin "lock-kernel-deps" ''
        #!/usr/bin/env bash
        set -euo pipefail

        kernel_dir="''${1:-.}"

        ${pkgs.lock-kernel-deps}/bin/lock-kernel-deps "$kernel_dir" -o "$kernel_dir/kernels.lock"
      '';

      kernels =
        pkgs.python3.withPackages (ps: [
          (overrideTorch ps.kernels)
        ])
        // {
          meta.mainProgram = "kernels";
        };

      # Package set by build variant, makes them easily available for tests,
      # ad-hoc shells, etc.
      pkgs = builtins.listToAttrs (
        map (buildSet: {
          name = variantOf buildSet;
          # `buildSet.pkgs` can be shared between multiple build sets and does
          # therefore not contain Torch. We override the Python packages with the
          # buildset-specific packages.
          value = buildSet.pkgs // {
            inherit (buildSet) python3;
            python3Packages = buildSet.python3.pkgs;
          };
        }) applicableBuildSets
      );
    }
    // lib.optionalAttrs hasRustKernels {
      inherit (pkgs) hash-rust;
    };
}
