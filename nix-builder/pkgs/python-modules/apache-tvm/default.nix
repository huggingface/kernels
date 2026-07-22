{
  lib,
  stdenv,
  fetchPypi,
  python,

  buildPythonPackage,
  autoAddDriverRunpath,
  autoPatchelfHook,
  pythonRelaxWheelDepsHook,
  pythonWheelDepsCheckHook,

  cudaPackages,

  ml-dtypes,
  numpy,
  typing-extensions,
  tvm-ffi,
}:

let
  format = "wheel";
  hashes = {
    x86_64-linux = "sha256-Z9H5XYYakxMMS6FWVkb760JZsqtHjqPyqdklZkiNlTA=";
    aarch64-linux = "sha256-ZHHNeh/QgDaToHxMLwsB9DNfvnrLMie1+dy8GxqKq0s=";
  };
  hash =
    hashes.${stdenv.system} or (throw "apache-tvm: unsupported system: ${stdenv.system}");
in
buildPythonPackage rec {
  pname = "apache-tvm";
  version = "0.25.0.post1";
  inherit format;

  src = fetchPypi {
    pname = "apache_tvm";
    dist = "py3";
    python = "py3";
    abi = "none";
    platform = "manylinux_2_27_${stdenv.hostPlatform.uname.processor}.manylinux_2_28_${stdenv.hostPlatform.uname.processor}";
    inherit format hash version;
  };

  nativeBuildInputs = [
    autoAddDriverRunpath
    autoPatchelfHook
    pythonRelaxWheelDepsHook
    pythonWheelDepsCheckHook
  ];

  buildInputs = [
    cudaPackages.cuda_cudart
  ];

  dependencies = [
    ml-dtypes
    numpy
    typing-extensions
    tvm-ffi
  ];

  # The wheel requires apache-tvm-ffi >= 0.1.12; our tvm-ffi build is used
  # as a compiler-side dependency only, so accept the pinned version.
  pythonRelaxDeps = [ "apache-tvm-ffi" ];

  # libtvm_ffi.so is inside the tvm-ffi Python package, which is not a
  # regular library path for auto-patchelf.
  preFixup = ''
    addAutoPatchelfSearchPath "${tvm-ffi}/${python.sitePackages}/tvm_ffi/lib"
  '';

  # The CUDA driver is provided by the host system at runtime. The wheel
  # links CUDA 13 cudart for kernel *execution*; we only use the package
  # for compilation (codegen), which does not load the CUDA runtime module.
  autoPatchelfIgnoreMissingDeps = [
    "libcuda.so.1"
    "libcudart.so.13"
  ];

  meta = {
    description = "Apache TVM ML compiler (binary wheel, provides the TIRx kernel DSL)";
    homepage = "https://tvm.apache.org";
    license = lib.licenses.asl20;
    broken = !stdenv.hostPlatform.isLinux;
    sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
  };
}
