{
  lib,
  autoPatchelfHook,
  callPackage,
  cpio,
  fetchurl,
  stdenv,
  rpm,
  rsync,
  rocmPackages,

  pname,
  version,

  # List of string-typed dependencies.
  deps,

  # List of derivations that must be merged.
  components,
}:

let
  filteredDeps = lib.filter (
    dep:
    !builtins.elem dep [
      "amdgpu-core"
      "libdrm-amdgpu-common"
      "libdrm-amdgpu-amdgpu1"
      "libdrm-amdgpu-radeon1"
      "libdrm-amdgpu-dev"
      "libdrm2-amdgpu"
    ]
  ) deps;
  srcs = map (component: fetchurl { inherit (component) url sha256; }) components;
in
stdenv.mkDerivation rec {
  inherit pname version srcs;

  nativeBuildInputs = [
    autoPatchelfHook
    rocmPackages.markForRocmRootHook
    cpio
    rpm
    rsync
  ];

  buildInputs = [
    stdenv.cc.cc.lib
    stdenv.cc.cc.libgcc
  ]
  ++ (map (dep: rocmPackages.${dep}) filteredDeps);

  # We do not use rpmextract anymore. It uses cpio, which fails to set
  # the setgid bit on directories. cpio will extract correctly, but return
  # an error, which causes rpmextract to fail. So instead we run cpio
  # directly and allow this particular error, but bail on other errors.
  unpackPhase = ''
    for src in $srcs; do
      if ! rpm2cpio "$src" | cpio --extract --make-directories --quiet 2>cpio.err; then
        if [ ! -s cpio.err ] || grep -v 'Cannot change mode' cpio.err >&2; then
          echo "unpacking $src failed" >&2
          exit 1
        fi
      fi
    done
    rm -f cpio.err
  '';

  installPhase = ''
    runHook preInstall
    mkdir $out
    cp -rT opt/rocm/core-* $out
    runHook postInstall
  '';

  # Stripping the binaries from the RHEL packages breaks them, so disable
  # it (seems kind of superfluous anyway, since the RPM build probably does
  # stripping as well).
  dontStrip = true;

  autoPatchelfIgnoreMissingDeps = [
    # Not sure where this comes from, not in the distribution.
    "amdpythonlib.so"

    # Should come from the driver runpath.
    "libOpenCL.so.1"

    # Distribution only has libamdhip64.so.6? Only seems to be used
    # by /bin/roofline-* for older Linux distributions.
    "libamdhip64.so.5"

    # Python versions not in nixpkgs anymore.
    "libpython3.6m.so.1.0"
    "libpython3.7m.so.1.0"
    "libpython3.8.so.1.0"
    "libpython3.9.so.1.0"
    "libpython3.10.so.1.0"
  ];
}
