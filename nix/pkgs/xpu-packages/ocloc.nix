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
  version = "25.35";

  srcs = [
    (fetchurl {
      url = "https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/intel-ocloc_25.35.35096.9-0_amd64.deb";
      hash = "sha256-AjzPj9iRKVi7PwCXcFbC+o3bzvjstLvHK9357LDuxvs=";
    })
    (fetchurl {
      url = "https://github.com/intel/intel-graphics-compiler/releases/download/v2.18.5/intel-igc-core-2_2.18.5+19820_amd64.deb";
      hash = "sha256-fDQPRcTylOyjqXgKGz7ObUtpn9xR59yf+MXWMjxdjWQ=";
    })
    (fetchurl {
      url = "https://github.com/intel/intel-graphics-compiler/releases/download/v2.18.5/intel-igc-opencl-2_2.18.5+19820_amd64.deb";
      hash = "sha256-vLFDVIYXVnPxSxgO+4/Dn/MbdsYl4JsSU6B6cF78+cM=";
    })
    (fetchurl {
      url = "https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/intel-opencl-icd_25.35.35096.9-0_amd64.deb";
      hash = "sha256-ifJSVTyTeyRW7pkZhY/yq2Xcaw3L0hnHOVNUhGWt8yU=";
    })
    (fetchurl {
      url = "https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/libigdgmm12_22.8.1_amd64.deb";
      hash = "sha256-gQEU7wX01ExchvvC8LNGiBAvQVlqQHcqH/ekj132ns0=";
    })
    (fetchurl {
      url = "https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/libze-intel-gpu1_25.35.35096.9-0_amd64.deb";
      hash = "sha256-phfUqwNRqPJxdDi+QyXKARB1hFDhDrb6AtYUrtElIHo=";
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
})
