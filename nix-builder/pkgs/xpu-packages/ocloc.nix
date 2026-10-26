{
  lib,
  stdenv,
  fetchurl,
  dpkg,
  autoPatchelfHook,
  ocloc,
  zlib,
  zstd,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "ocloc";
  version = "26.22";

  srcs = [
    (fetchurl {
      url = "https://github.com/intel/compute-runtime/releases/download/26.22.38646.4/intel-ocloc_26.22.38646.4-0_amd64.deb";
      hash = "sha256-JXRYQeZiecf/H/bC1NqeyRFoXBLhycVgnZnk1Uo2TcA=";
    })
    (fetchurl {
      url = "https://github.com/intel/intel-graphics-compiler/releases/download/v2.36.3/intel-igc-core-2_2.36.3+21719_amd64.deb";
      hash = "sha256-ngl1rHUBW0Meuy2oGoArn9HiijwnAxOpdWnNHmpsYEg=";
    })
    (fetchurl {
      url = "https://github.com/intel/intel-graphics-compiler/releases/download/v2.36.3/intel-igc-opencl-2_2.36.3+21719_amd64.deb";
      hash = "sha256-NQpSMx54S7f7ntQumTtcRLfmVi/HTSzzECsptqV2+oU=";
    })
    (fetchurl {
      url = "https://github.com/intel/compute-runtime/releases/download/26.22.38646.4/intel-opencl-icd_26.22.38646.4-0_amd64.deb";
      hash = "sha256-b9rC6KKqz4ROv9kFIb9xArPrtE9pwbztGpeFp86Wo8I=";
    })
    (fetchurl {
      url = "https://github.com/intel/compute-runtime/releases/download/26.22.38646.4/libigdgmm12_22.10.0_amd64.deb";
      hash = "sha256-YDGmPW6KEs5hwU78FfLI5ycGEobjgguFlObQBhXgTVQ=";
    })
    (fetchurl {
      url = "https://github.com/intel/compute-runtime/releases/download/26.22.38646.4/libze-intel-gpu1_26.22.38646.4-0_amd64.deb";
      hash = "sha256-i++fJOA/gm+TwHYIG9oTxqw6+9nkK5+48pj6tlIzDi8=";
    })
  ];
  dontStrip = true;

  nativeBuildInputs = [
    dpkg
    autoPatchelfHook
  ];

  buildInputs = [
    stdenv.cc.cc.lib
    zlib
    zstd
  ];

  unpackPhase = ''
    for src in $srcs; do
      dpkg-deb -x "$src" .
    done
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin $out/lib
    find . -name 'ocloc*' -exec cp {} $out/bin/ \;
    find . -name '*.so*' -exec cp {} $out/lib/ \;
    mv $out/bin/ocloc-${finalAttrs.version}* $out/bin/ocloc
    runHook postInstall
  '';

  # Some libraries like libigc.so are dlopen'ed from other shared
  # libraries in the package. So we need to add the library path
  # to RPATH. Ideally we'd want to use
  #
  # runtimeDependencies = [ (placeholder "out") ];
  #
  # But it only adds the dependency to binaries, not shared
  # libraries, so we hack around it here.
  doInstallCheck = true;
  preInstallCheck = ''
    patchelf --add-rpath ${placeholder "out"}/lib $out/lib/*.so*
  '';

  meta = with lib; {
    description = "Intel OpenCL Offline Compiler";
    homepage = "https://github.com/intel/compute-runtime";
    platforms = platforms.linux;
    license = licenses.mit;
    sourceProvenance = with sourceTypes; [ binaryNativeCode ];
  };
})
