{
  buildPythonPackage,
  fetchFromGitHub,
  fetchpatch,

  scikit-build-core,
  setuptools,

  cmake,
  ninja,

  jax,
  tvm-ffi,
  typing-extensions,
}:

buildPythonPackage rec {
  pname = "jax-tvm-ffi";
  version = "0.1.2";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = pname;
    tag = "v${version}";
    hash = "sha256-TRZP3CYQls6Q0MEhOA+YfBVN+BSI7C63YwL7CId/nfE=";
  };

  patches = [
    # Compatibility with tvm-ffi >= 0.1.8.post2.
    (fetchpatch {
      url = "https://github.com/NVIDIA/jax-tvm-ffi/commit/e35941de1b51b1655cbfcc7d081035000aa408bd.diff";
      hash = "sha256-nu0XMjl4UB9fqCi/jk/jAyyp1wUPiclvr6LBnI8/nD8=";
    })
  ];

  build-system = [
    cmake
    ninja
    scikit-build-core
  ];

  dontUseCmakeConfigure = true;

  dependencies = [
    jax
    tvm-ffi
    typing-extensions
  ];

  pythonRelaxDeps = [ "apache-tvm-ffi" ];
}
