{
  stdenv,
  lib,
  buildPythonPackage,
  fetchPypi,
  python,

  autoAddDriverRunpath,
  autoPatchelfHook,
  pythonWheelDepsCheckHook,

  cudaPackages,
  cuda-bindings,
  cuda-pathfinder,
  numpy,
  pyelftools,
}:

let
  version = "13.4.0";

  hashes = {
    x86_64-linux = "sha256-OMr/i6qbr868gtGwSmr1jSwBlCCPhvavE/SkmPWP8mk=";
    aarch64-linux = "sha256-aDCxEjJTby+9salK8uX+hltDbRKVLhgFDeFGpki0jZI=";
  };

  hash =
    hashes.${stdenv.hostPlatform.system}
      or (throw "No hash defined for system: ${stdenv.hostPlatform.system}");

  format = "wheel";
  pyShortVersion = "cp" + builtins.replaceStrings [ "." ] [ "" ] python.pythonVersion;
in
buildPythonPackage {
  inherit format version;
  pname = "cupti-python";

  src = fetchPypi {
    pname = "cupti_python";
    python = pyShortVersion;
    abi = pyShortVersion;
    dist = pyShortVersion;
    platform = "manylinux_2_24_${stdenv.hostPlatform.uname.processor}.manylinux_2_28_${stdenv.hostPlatform.uname.processor}";
    inherit format hash version;
  };

  nativeBuildInputs = [
    autoPatchelfHook
    # After autoPatchelfHook to avoid overriding.
    autoAddDriverRunpath
    pythonWheelDepsCheckHook
  ];

  # We use libcupti from the Nix package.
  pythonRemoveDeps = [ "nvidia-cuda-cupti" ];

  buildInputs = [
    stdenv.cc.cc.lib
    cudaPackages.cuda_cupti
  ];

  # This package uses libcupti, but it relies on cuda-pathfinder to
  # dynamically load it. Add an explicit dependency, so that it can
  # be found.
  postInstall = ''
    patchelf --add-needed libcupti.so.13 \
      $out/${python.sitePackages}/cupti/_internal/cupti.*.so
  '';

  dependencies = [
    cuda-bindings
    cuda-pathfinder
    numpy
    pyelftools
  ];

  pythonImportsCheck = [ "cupti" ];

  meta = {
    description = "Python bindings for the CUDA Profiling Tools Interface (CUPTI)";
    homepage = "https://github.com/NVIDIA/cupti-python";
    license = lib.licenses.unfreeRedistributable;
    sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
  };
}
