{
  lib,
  fetchurl,
  rpmextract,
  stdenvNoCC,

  autoPatchelfHook,
  fakeroot,

  manylinuxPackages,

  pname,
  version,

  # List of string-typed dependencies.
  deps,

  # List of derivations that must be merged.
  components,
}:

let
  srcs = map (component: fetchurl { inherit (component) url sha256; }) components;
in
stdenvNoCC.mkDerivation (
  finalAttrs:
  let
    filteredDeps = lib.filter (dep: !builtins.elem dep finalAttrs.passthru.filterDeps) deps;
  in
  {
    inherit pname version srcs;

    nativeBuildInputs = [
      autoPatchelfHook
      fakeroot
      rpmextract
    ];

    buildInputs = map (dep: manylinuxPackages.${dep}) filteredDeps;

    # Extract RPM packages using rpmextract. The package set has some
    # setuid/setgid binaries. Use fakeroot to avoid extraction errors.
    unpackPhase = ''
      for src in $srcs; do
        fakeroot rpmextract "$src"
      done
    '';

    installPhase = ''
      runHook preInstall

      mkdir -p $out

      if [ -d opt/rh/gcc-toolset-13/root ]; then
        root=opt/rh/gcc-toolset-13/root
      elif [ -d opt/rh/gcc-toolset-14/root ]; then
        root=opt/rh/gcc-toolset-14/root
      else
        root=.
      fi

      for d in bin include lib lib64 libexec sbin usr/bin usr/include usr/lib usr/lib64 usr/libexec usr/sbin; do
        if [ -d "$root/$d" -a ! -L "$root/$d" ]; then
          cp -r $root/$d $out/
        fi
      done

      rm -rf $out/lib/.build-id

      runHook postInstall
    '';

    dontStrip = true;

    passthru = {
      # Dependencies to filter out, by their (string-typed) names.
      filterDeps = [ "filesystem" ];
    };

    meta = with lib; {
      description = "AlmaLinux package for manylinux: ${pname}";
      homepage = "https://almalinux.org";
      platforms = platforms.linux;
      sourceProvenance = with sourceTypes; [
        binaryNativeCode
      ];
    };
  }
)
