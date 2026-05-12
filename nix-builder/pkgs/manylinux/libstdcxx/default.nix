{
  fetchurl,
  rpmextract,
  stdenvNoCC,

  autoPatchelfHook,

  almalinux-glibc
}:

# TODO: make generic (multiple almalinux versions), add aarch64

stdenvNoCC.mkDerivation (finalAttrs: {
  pname = "libstdc++";
  version = "8.5.0-28.el8_10.alma.1";

  src = fetchurl {
    url = "http://mirror.transip.net/almalinux/8/BaseOS/x86_64/os/Packages/libstdc%2B%2B-${finalAttrs.version}.x86_64.rpm";
          #http://mirror.transip.net/almalinux/8/BaseOS/x86_64/os/Packages/libstdc%2B%2B-8.5.0-28.el8_10.alma.1.x86_64.rpm
    hash = "sha256-AwK5AG1xmjHdUa6vMf7APXCq1cMYZ0U3vYDEn5F0sGY=";
  };

  buildInputs = [
    almalinux-glibc
  ];

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


