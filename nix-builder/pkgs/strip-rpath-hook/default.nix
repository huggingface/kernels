{
  lib,
  stdenv,
  makeSetupHook,

  patchelf,
  rewrite-nix-paths-macho,
}:

# Setup hook that makes shared libraries portable by stripping the RPATHs
# that point to the Nix store. This is done in `postInstall` (after the
# derivation's own `postInstall`), so before the fixup phase and before any
# install checks.

makeSetupHook {
  name = "strip-rpath-hook";
  substitutions = {
    patchelf = lib.optionalString stdenv.hostPlatform.isLinux "${patchelf}/bin/patchelf";
    rewrite_nix_paths_macho = lib.optionalString stdenv.hostPlatform.isDarwin "${rewrite-nix-paths-macho}/bin/rewrite-nix-paths-macho";
  };
} ./strip-rpath-hook.sh
