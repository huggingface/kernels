{
  lib,
  stdenvNoCC,
}:

stdenvNoCC.mkDerivation {
  name = "torch-stable-abi-shim";

  src = ../../../torch-stable-abi-shim;

  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/include"
    cp -r include/. "$out/include/"

    runHook postInstall
  '';

  meta = {
    description = "Header-only shim for the PyTorch stable ABI";
    platforms = lib.platforms.all;
  };
}
