{
  lib,
  fetchurl,

  buildPythonPackage,
  autoPatchelfHook,
  autoAddDriverRunpath,
  pythonWheelDepsCheckHook,

  zlib,

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

  buildInputs = [ zlib ];

  dependencies = [
    pyelftools
  ];

  dontStrip = true;

  pythonImportsCheck = [ "triton" ];

  meta = with lib; {
    description = "GPU language and compiler";
    homepage = "https://github.com/triton-lang/triton";
    license = lib.licenses.mit;
    sourceProvenance = with sourceTypes; [ binaryNativeCode ];
  };
}
