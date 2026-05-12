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
  filters = {
    glibc = [
      "glibc-common"
      "libxcrypt"
    ];
    glibc-common = [
      "glibc"
      "libselinux"
    ];
    glibc-headers = [ "glibc" ];
    glibc-minimal-langpack = [
      "glibc"
      "glibc-common"
    ];
    platform-python-setuptools = [ "platform-python" ];
    python3-libs = [ "platform-python" ];
    rpm-libs = [ "rpm" ];
  };
  filteredDeps = lib.filter (
    dep: dep != "filesystem" && !builtins.elem dep (filters.${pname} or [ ])
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

    mkdir -p $out

    if [ -d opt/rh/gcc-toolset-14/root ]; then
      root=opt/rh/gcc-toolset-14/root
    else
      root=.
    fi

    for d in bin lib lib64 libexec sbin usr/bin usr/lib usr/lib64 usr/libexec usr/sbin; do
      if [ -d "$root/$d" -a ! -L "$root/$d" ]; then
        cp -r $root/$d $out/
      fi
    done

    rm -rf $out/lib/.build-id

    runHook postInstall
  '';

  dontStrip = true;
})
