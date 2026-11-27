#!/bin/sh

echo "Sourcing strip-rpath-hook.sh"

# Strip all rpaths.
_stripRPathLinux() {
  find "$out/" -name '*.so' \
    -exec @patchelf@ --set-rpath "" {} \;
}

# Strip all rpaths on Darwin.
_stripRPathDarwin() {
  find "$out/" -name '*.so' \
    -exec @rewrite_nix_paths_macho@ {} \;

  # Stub some rpath.
  find "$out/" -name '*.so' \
    -exec install_name_tool -add_rpath "@loader_path/lib" {} \;
}

# Strip all rpaths, but keep origin-relative ones.
_stripRPathKeepOriginLinux() {
  local lib rpath kept entry
  local -a entries

  while IFS= read -r -d '' lib; do
    rpath=$(@patchelf@ --print-rpath "$lib" 2>/dev/null) || continue
    [ -n "$rpath" ] || continue

    kept=""
    IFS=':' read -ra entries <<<"$rpath"
    for entry in "${entries[@]}"; do
      case "$entry" in
        '$ORIGIN' | '$ORIGIN/'* | '${ORIGIN}' | '${ORIGIN}/'*)
          kept="${kept:+$kept:}$entry"
          ;;
        *)
          echo "  ${lib#"$out"/}: removing RPATH entry '$entry'"
          ;;
      esac
    done

    if [ "$kept" != "$rpath" ]; then
      @patchelf@ --set-rpath "$kept" "$lib"
    fi
  done < <(find "$out/" -name '*.so' -print0)
}

# Strip all rpaths on Darwin, but keep origin-relative ones.
_stripRPathKeepOriginDarwin() {
  local lib rpath

  while IFS= read -r -d '' lib; do
    while IFS= read -r rpath; do
      case "$rpath" in
        '@loader_path'* | '@executable_path'*) ;;
        *)
          echo "  ${lib#"$out"/}: removing LC_RPATH entry '$rpath'"
          install_name_tool -delete_rpath "$rpath" "$lib"
          ;;
      esac
    done < <(otool -l "$lib" | awk '/cmd LC_RPATH/ { f = 1 } f && $1 == "path" { print $2; f = 0 }')
  done < <(find "$out/" -name '*.so' -print0)
}

stripRPathHook() {
  echo "Stripping RPATHs from shared libraries"

  if [ -n "${stripRPathKeepOrigin-}" ]; then
    echo "Keeping origin-relative RPATH entries"

    if [ -n "@patchelf@" ]; then
      _stripRPathKeepOriginLinux
    elif [ -n "@rewrite_nix_paths_macho@" ]; then
      _stripRPathKeepOriginDarwin
    fi
  else
    if [ -n "@patchelf@" ]; then
      _stripRPathLinux
    elif [ -n "@rewrite_nix_paths_macho@" ]; then
      _stripRPathDarwin
    fi
  fi
}

if [ -z "${dontStripRPath-}" ]; then
  postInstallHooks+=(stripRPathHook)
fi
