{
  autoPatchelfHook,
  clr,
  fetchurl,
  lib,
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
  buildInputs = [
    clr
    stdenv.cc.cc.lib
    xz
  ];

  dontConfigure = true;
  dontBuild = true;
  dontStrip = true;
  installPhase = ''
    runHook preInstall

    mkdir -p "$out"
    cp -r include lib "$out/"
    ln -s ${images}/lib/aotriton.images "$out/lib/aotriton.images"

    runHook postInstall
  '';

  meta = with lib; {
    description = "Ahead of Time (AOT) Triton Math Library";
    homepage = "https://github.com/ROCm/aotriton";
    license = with licenses; [ mit ];
    platforms = [ "x86_64-linux" ];
    sourceProvenance = with sourceTypes; [ binaryNativeCode ];
  };
}
