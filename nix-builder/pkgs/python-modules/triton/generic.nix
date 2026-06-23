{
  lib,
  fetchurl,

  python,
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

  postInstall =
    let
      # `libcuda_dirs` in Triton probes for libcuda.so.1 using ldconfig. We
      # cannot use ldconfig on Nix and we know where libcuda.so.1 is, so just
      # hardcode it.
      replacement = ''
        def libcuda_dirs():
            return ['/run/opengl-driver/lib']

        def _libcuda_dirs():
      '';
    in
    ''
      substituteInPlace $out/${python.sitePackages}/triton/backends/nvidia/driver.py \
        --replace-fail "def libcuda_dirs():" ${lib.escapeShellArg replacement}
    '';

  dontStrip = true;

  pythonImportsCheck = [ "triton" ];

  meta = with lib; {
    description = "GPU language and compiler";
    homepage = "https://github.com/triton-lang/triton";
    license = lib.licenses.mit;
    sourceProvenance = with sourceTypes; [ binaryNativeCode ];
  };
}
