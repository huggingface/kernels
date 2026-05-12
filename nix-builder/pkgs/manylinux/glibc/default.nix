{
  fetchurl,
  rpmextract,
  stdenvNoCC,

  autoPatchelfHook,
}:

# TODO: make generic (multiple almalinux versions), add aarch64

stdenvNoCC.mkDerivation (finalAttrs: {
  pname = "glibc";
  version = "2.28-251.el8_10.34";

  src = fetchurl {
    url = "http://mirror.transip.net/almalinux/8/BaseOS/x86_64/os/Packages/glibc-${finalAttrs.version}.x86_64.rpm";
    hash = "sha256-IWeHdVV1xkurDHd7mZQYU+ldIvYtPalOpobeFgZ6LYs=";
  };

  nativeBuildInputs = [
    autoPatchelfHook
    rpmextract
  ];

  unpackPhase = ''
    rpmextract "$src"
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out
    find usr \
      -type d \
      \( -name "bin" -o -name "include" -o -name "lib" -o -name "libexec" -o -name "share" \) \
      -prune \
      -exec cp -r {} $out/ \;
    rm -rf $out/lib/.build-id
    cp -r usr/lib64/* $out/lib

    runHook postInstall
  '';
})
