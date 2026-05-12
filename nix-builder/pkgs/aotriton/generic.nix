{
  autoPatchelfHook,
  clr,
  fetchurl,
  lib,
  python3,
  rocm-core,
  stdenv,
  xz,
}:

{
  version,
  images,
  hashes,
}:

let
  rocmVersion = lib.versions.majorMinor rocm-core.version;
  hash =
    hashes.${rocmVersion}
      or (throw "aotriton ${version} binary package is not specified for ROCm ${rocmVersion}");
in
stdenv.mkDerivation {
  pname = "aotriton";
  inherit version;

  src = fetchurl {
    url = "https://github.com/ROCm/aotriton/releases/download/${version}/aotriton-${version}-manylinux_2_28_x86_64-rocm${rocmVersion}-shared.tar.gz";
    inherit hash;
  };

  nativeBuildInputs = [ autoPatchelfHook ];
  nativeInstallCheckInputs = [ python3 ];

  buildInputs = [
    clr
    stdenv.cc.cc.lib
    xz
  ];

  dontConfigure = true;
  dontBuild = true;
  dontStrip = true;
  doInstallCheck = true;

  installPhase = ''
    runHook preInstall

    mkdir -p "$out"
    cp -r include lib "$out/"
    ln -s ${images}/lib/aotriton.images "$out/lib/aotriton.images"

    runHook postInstall
  '';

  installCheckPhase = ''
    runHook preInstallCheck

    test -f "$out/lib/libaotriton_v2.so"
    test -d "$out/lib/aotriton.images"
    python - <<PY
    import ctypes
    import os

    ctypes.CDLL(
        "$out/lib/libaotriton_v2.so",
        mode=os.RTLD_NOW | os.RTLD_LOCAL,
    )
    PY

    runHook postInstallCheck
  '';

  meta = with lib; {
    description = "Ahead of Time (AOT) Triton Math Library";
    homepage = "https://github.com/ROCm/aotriton";
    license = with licenses; [ mit ];
    platforms = [ "x86_64-linux" ];
    sourceProvenance = with sourceTypes; [ binaryNativeCode ];
  };
}
