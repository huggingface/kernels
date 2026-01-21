{
  lib,
  autoPatchelfHook,
  callPackage,
  fetchurl,
  stdenv,
  rpmextract,
  rsync,
  zlib,

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
stdenv.mkDerivation rec {
  inherit pname version srcs;

  nativeBuildInputs = [
    autoPatchelfHook
    rpmextract
    rsync
  ];

  buildInputs = [
    stdenv.cc.cc.lib
    stdenv.cc.cc.libgcc
    zlib
  ];

  # Extract RPM packages using rpmextract
  unpackPhase = ''
    for src in $srcs; do
      rpmextract "$src"
    done
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out
    find . -type d \( -name "bin" -o -name "include" -o -name "lib" -o -name "share" \) -prune -exec cp -r {} $out/ \;

    runHook postInstall
  '';

  # Stripping the binaries from the oneAPI packages might break them
  dontStrip = true;

  # Don't check for broken symlinks - Intel packages often have complex internal symlink structures
  preFixup = ''
    # Remove broken symlinks that point to Intel's "latest" structure
    find $out -type l ! -exec test -e {} \; -delete 2>/dev/null || true
  '';

  autoPatchelfIgnoreMissingDeps = [
    # oneAPI specific libraries that should come from driver/runtime
    "libOpenCL.so.1"
    "libze_loader.so.1"
    "libtbbmalloc.so.2"
    "libtbb.so.12"
    "libsycl.so.8" # Intel SYCL runtime library
    "libmkl_sycl_blas.so.5"
    "libhwloc.so.15" # Hardware Locality library
    "libhwloc.so.5" # Hardware Locality library

    # System libraries that might not be available
    "libpython3.6m.so.1.0"
    "libpython3.7m.so.1.0"
    "libpython3.8.so.1.0"
    "libpython3.9.so.1.0"
  ];

  meta = with lib; {
    description = "Intel oneAPI package: ${pname}";
    homepage = "https://software.intel.com/oneapi";
    platforms = platforms.linux;
    license = licenses.unfree;
  };
}
