{
  lib,
  stdenv,
  buildPythonPackage,
  fetchurl,

  autoPatchelfHook,
  pythonRelaxWheelDepsHook,
  pythonWheelDepsCheckHook,
  python,

  aws-neuronx-runtime-lib,
  bzip2,
  libarchive,
  openssl,
  xz,
  zlib,

  ec2-metadata,
  islpy,
  ml-dtypes,
  networkx,
  nki,
  numpy,
  protobuf,
  psutil,
  python-daemon,
  requests-unixsocket,
  scipy,
  tqdm,
  typing-extensions,
}:

let
  version = "2.28.7631.0+8f7b0cf5";

  # Only the CPython 3.14 wheel is supported.
  abi = "cp314";
  filename = "neuronx_cc-${version}-${abi}-${abi}-linux_x86_64.whl";
in
assert lib.assertMsg (
  python.pythonVersion == "3.14"
) "neuronx-cc: unsupported Python version: ${python.pythonVersion}, only 3.14 is supported";
buildPythonPackage {
  pname = "neuronx-cc";
  inherit version;
  format = "wheel";

  src = fetchurl {
    # The local version separator needs to be escaped in URLs.
    url = "https://pip.repos.neuron.amazonaws.com/neuronx-cc/${
      lib.replaceStrings [ "+" ] [ "%2B" ] filename
    }";
    # The store path name cannot contain `%`, but the wheel file name has to
    # be preserved for wheel installation.
    name = filename;
    hash = "sha256-C91pjm6pTZi3a0fRCNHVZQ0UKE1mQgFPNFNn24t34FQ=";
  };

  # We use our own wheel dependency check.
  dontCheckRuntimeDeps = true;

  nativeBuildInputs = [
    autoPatchelfHook
    pythonRelaxWheelDepsHook
    pythonWheelDepsCheckHook
  ];

  buildInputs = [
    aws-neuronx-runtime-lib
    bzip2
    libarchive
    openssl
    stdenv.cc.cc.lib
    xz
    zlib
  ];

  # The wheel is ~1.9 GB, mostly large prebuilt binaries. Stripping them
  # takes a long time and is not worth it.
  dontStrip = true;

  dependencies = [
    ec2-metadata
    islpy
    ml-dtypes
    networkx
    nki
    numpy
    protobuf
    psutil
    python-daemon
    requests-unixsocket
    scipy
    tqdm
    typing-extensions
  ];

  # The wheel pins networkx~=2.6 and ec2-metadata~=2.10.
  pythonRelaxWheelDeps = [
    "ec2-metadata"
    "networkx"
  ];

  # Declared, but nothing in the wheel uses it.
  pythonRemoveWheelDeps = [ "pgzip" ];

  pythonImportsCheck = [ "neuronxcc" ];

  meta = {
    description = "AWS Neuron SDK compiler";
    homepage = "https://aws.amazon.com/machine-learning/inferentia/";
    # AWS Neuron Software License Agreement, which prohibits redistribution.
    license = lib.licenses.unfree;
    mainProgram = "neuronx-cc";
    platforms = [ "x86_64-linux" ];
    sourceProvenance = [ lib.sourceTypes.binaryNativeCode ];
  };
}
