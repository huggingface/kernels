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
  pname = "triton-xpu";
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
    description = "Triton XPU backend for PyTorch";
    homepage = "https://github.com/intel/intel-xpu-backend-for-triton";
    license = lib.licenses.mit;
    sourceProvenance = with sourceTypes; [ binaryNativeCode ];
  };
}
