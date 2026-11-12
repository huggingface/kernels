{
  lib,
  stdenv,
  buildPythonPackage,
  fetchurl,

  autoPatchelfHook,
  python,
  zlib,
  zstd,

  ml-dtypes,
  numpy,
  typing-extensions,
}:

let
  version = "0.7.0+37713544100.gfede04bb";

  # Only the CPython 3.14 wheel is supported.
  abi = "cp314";
  filename = "nki-${version}-${abi}-${abi}-linux_x86_64.whl";
in
assert lib.assertMsg (
  python.pythonVersion == "3.14"
) "nki: unsupported Python version: ${python.pythonVersion}, only 3.14 is supported";
buildPythonPackage {
  pname = "nki";
  inherit version;
  format = "wheel";

  src = fetchurl {
    # The local version separator needs to be escaped in URLs.
    url = "https://pip.repos.neuron.amazonaws.com/nki/${lib.replaceStrings [ "+" ] [ "%2B" ] filename}";
    # The store path name cannot contain `%`, but the wheel file name has to
    # be preserved for wheel installation.
    name = filename;
    hash = "sha256-Fbp91hqozYItvPwUrJ2bOTKkYZqZdnkUzh44Lvpjh0Q=";
  };

  nativeBuildInputs = [ autoPatchelfHook ];

  buildInputs = [
    stdenv.cc.cc.lib
    zlib
    zstd
  ];

  dependencies = [
    ml-dtypes
    numpy
    typing-extensions
  ];

  pythonImportsCheck = [ "nki" ];

  meta = {
    description = "Neuron Kernel Interface (NKI), MLIR-based kernel compiler for Neuron devices";
    homepage = "https://github.com/aws-neuron/nki";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
    sourceProvenance = [ lib.sourceTypes.binaryNativeCode ];
  };
}
