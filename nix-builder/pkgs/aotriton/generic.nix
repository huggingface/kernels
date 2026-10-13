{
  autoPatchelfHook,
  fetchurl,
  lib,
  rocmPackages,
  stdenv,
  xz,
}:

{
  version,
  images,
  hashes,

  # Maps a ROCm version to the HIP version that upstream uses in the
  # binary package name, for ROCm versions where they differ.
  hipVersions ? { },
}:

let
  rocmVersion = lib.versions.majorMinor rocmPackages.version;
  hipVersion = hipVersions.${rocmVersion} or rocmVersion;
  hash =
    hashes.${rocmVersion}
      or (throw "aotriton ${version} binary package is not specified for ROCm ${rocmVersion}");
in
stdenv.mkDerivation {
  pname = "aotriton";
  inherit version;

  src = fetchurl {
    url = "https://github.com/ROCm/aotriton/releases/download/${version}/aotriton-${version}-manylinux_2_28_x86_64-rocm${hipVersion}-shared.tar.gz";
    inherit hash;
  };

  nativeBuildInputs = [ autoPatchelfHook ];
  buildInputs = [
    stdenv.cc.cc.lib
    xz
    rocmPackages.clr
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
