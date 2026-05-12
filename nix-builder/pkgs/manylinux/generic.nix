{
  lib,
  fetchurl,
  rpmextract,
  stdenvNoCC,

  autoPatchelfHook,

  manylinuxPackages,

  pname,
  version,

  # List of string-typed dependencies.
  deps,

  # List of derivations that must be merged.
  components,
}:

# TODO: make generic (multiple almalinux versions), add aarch64

let
  filteredDeps = lib.filter (
    dep:
    !builtins.elem dep [
      # Break glibc -> glibc-common -> glibc cycle.
      "filesystem"
      "glibc-common"
      "glibc"
    ]
  ) deps;
  srcs = map (component: fetchurl { inherit (component) url sha256; }) components;
in
stdenvNoCC.mkDerivation (finalAttrs: {
  inherit pname version srcs;

  nativeBuildInputs = [
    autoPatchelfHook
    rpmextract
  ];

  buildInputs = map (dep: manylinuxPackages.${dep}) filteredDeps;

  # Extract RPM packages using rpmextract
  unpackPhase = ''
    echo "sources: $srcs"
    for src in $srcs; do
      rpmextract "$src"
    done
  '';

  installPhase = ''
    runHook preInstall

    find

    echo "creating..."

    mkdir -p $out
    for d in bin lib sbin usr/bin usr/lib usr/sbin; do
      echo $d
      if [ -d "$d" ]; then
        cp -r $d $out/
      fi
    done

    for d in lib64 usr/lib64; do
      echo $d
      if [ -d "$d" ]; then
        cp -r $d $out/
      fi
    done

    rm -rf $out/lib/.build-id

    find $out



    runHook postInstall
  '';

  dontStrip = true;
})


