#!/bin/sh

echo "Sourcing strip-rpath-hook.sh"

stripRPathHook() {
  echo "Stripping RPATHs from shared libraries"

  if [ -n "@patchelf@" ]; then
    find "$out/" -name '*.so' \
      -exec @patchelf@ --set-rpath "" {} \;
  fi

  if [ -n "@rewrite_nix_paths_macho@" ]; then
    find "$out/" -name '*.so' \
      -exec @rewrite_nix_paths_macho@ {} \;

    # Stub some rpath.
    find "$out/" -name '*.so' \
      -exec install_name_tool -add_rpath "@loader_path/lib" {} \;
  fi
}

if [ -z "${dontStripRPath-}" ]; then
  postInstallHooks+=(stripRPathHook)
fi
