{
  lib,
  stdenv,
  fetchPypi,
  buildPythonPackage,
  autoPatchelfHook,

  absl-py,
  cheroot,
  duckdb,
  etils,
  fire,
  fsspec,
  gcsfs,
  google-cloud-storage,
  gviz-api,
  ml-dtypes,
  numpy,
  protobuf,
  six,
  urllib3,
  werkzeug,
}:

buildPythonPackage rec {
  pname = "xprof";
  version = "2.24.2";
  format = "wheel";

  src = fetchPypi {
    inherit format pname version;
    dist = "py3";
    python = "py3";
    abi = "none";
    platform = "manylinux_2_27_x86_64";
    hash = "sha256-FDQmdMU87eKkwXxOIyep50nzlxeVwkdU4OGypjiTXlU=";
  };

  nativeBuildInputs = [ autoPatchelfHook ];

  buildInputs = [ stdenv.cc.cc.lib ];

  dependencies = [
    absl-py
    cheroot
    duckdb
    etils
    fire
    fsspec
    # fsspec[gcs]
    gcsfs
    google-cloud-storage
    gviz-api
    ml-dtypes
    numpy
    protobuf
    six
    urllib3
    werkzeug
  ]
  ++ etils.optional-dependencies.epath;

  pythonImportsCheck = [ "xprof" ];

  meta = with lib; {
    description = "Profiler for XLA programs (JAX, TensorFlow, PyTorch/XLA)";
    homepage = "https://github.com/openxla/xprof";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
