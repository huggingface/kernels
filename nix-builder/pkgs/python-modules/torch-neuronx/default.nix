{
  lib,
  stdenv,
  buildPythonPackage,
  fetchurl,

  autoPatchelfHook,
  aws-neuronx-runtime-lib,
  python,
  zlib,

  grpcio-tools,
  neuronx-cc,
  numpy,
  packaging,
  torch,
  wheel,
}:

let
  release = "v2.33.0";

  # Only the CPython 3.14 wheels are supported. The wheels are per Torch
  # major/minor version. The version contains a build number, so the file
  # names cannot be derived from the Torch version alone.
  abi = "cp314";

  wheels = {
    "2.13" = {
      version = "2.13.3.0.284+dc70ffa";
      hash = "sha256-b2SwyzalO3I8IOduUYbXyoanS5Az05s4aEGrPN3O78U=";
    };
    "2.14" = {
      version = "2.14.3.0.282+dc70ffa";
      hash = "sha256-6KGR0jBledRhBgp/HchRGmuzcOAetgWdyRdzbiUjapg=";
    };
  };

  torchMajorMinor = lib.versions.majorMinor torch.version;

  inherit
    (
      if python.pythonVersion == "3.14" then
        wheels.${torchMajorMinor} or (throw "torch-neuronx: unsupported Torch version: ${torch.version}")
      else
        throw "torch-neuronx: unsupported Python version: ${python.pythonVersion}, only 3.14 is supported"
    )
    version
    hash
    ;

  # Wheel file names use an underscore.
  filename = "torch_neuronx-${version}-${abi}-${abi}-linux_x86_64.whl";
in
buildPythonPackage {
  pname = "torch-neuronx";
  inherit version;
  format = "wheel";

  src = fetchurl {
    # The local version separator needs to be escaped in URLs.
    url = "https://github.com/aws-neuron/torch-neuronx/releases/download/${release}/${
      lib.replaceStrings [ "+" ] [ "%2B" ] filename
    }";
    # The store path name cannot contain `%`, but the wheel file name has to
    # be preserved for wheel installation.
    name = filename;
    inherit hash;
  };

  nativeBuildInputs = [ autoPatchelfHook ];

  buildInputs = [
    aws-neuronx-runtime-lib
    stdenv.cc.cc.lib
    zlib
  ];

  # The libtorch libraries are not in a `lib` directory that
  # autoPatchelfHook searches by default.
  preFixup = ''
    addAutoPatchelfSearchPath "${torch}/${python.sitePackages}/torch/lib"
  '';

  dependencies = [
    grpcio-tools
    neuronx-cc
    numpy
    packaging
    torch
    wheel
  ];

  # Importing `torch_neuronx` queries the Neuron instance, which fails
  # without a Neuron device. Override the queried values, so that the
  # import check can run in the build sandbox.
  env = {
    NEURON_PLATFORM_TARGET_OVERRIDE = "trn2";
    NEURON_LOGICAL_NC_CONFIG = "2";
  };

  pythonImportsCheck = [ "torch_neuronx" ];

  meta = with lib; {
    description = "Neuron backend for PyTorch";
    homepage = "https://github.com/aws-neuron/torch-neuronx";
    license = licenses.asl20;
    sourceProvenance = with sourceTypes; [ binaryNativeCode ];
    platforms = [ "x86_64-linux" ];
  };
}
