{
  lib,
  fetchurl,

  buildPythonPackage,
  autoPatchelfHook,
  autoAddDriverRunpath,
  pythonWheelDepsCheckHook,

  pkgs,

  pyelftools,
}:

{
  version,
  url,
  hash,
}:
buildPythonPackage {
  pname = "triton";
  inherit version;
  format = "wheel";

  src = fetchurl {
    inherit url hash;
  };

  nativeBuildInputs = [
    autoPatchelfHook
    pythonWheelDepsCheckHook
  ];

  buildInputs = with pkgs; [
    bzip2
    xz
    zlib
    zstd
  ];

  dependencies = [
    pyelftools
  ];

  dontStrip = true;

  pythonImportsCheck = [ "triton" ];

  meta = with lib; {
    description = "GPU language and compiler (ROCm)";
    homepage = "https://github.com/triton-lang/triton";
    license = lib.licenses.mit;
    sourceProvenance = with sourceTypes; [ binaryNativeCode ];
  };
}
