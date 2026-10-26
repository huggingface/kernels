{
  lib,
  fetchPypi,
  buildPythonPackage,
  requireFile,
  python,
}:

let
  # The wheel ships per-CPython-ABI builds (cp311..cp314); pick the tag
  # matching the python this package set is built for. The hash below is
  # for cp314 (the nixpkgs default python); if either moves, recompute
  # via the prefetch script above.
  abi = "cp${lib.versions.major python.version}${lib.versions.minor python.version}";
in
buildPythonPackage rec {
  pname = "libtpu";
  version = "0.0.50";
  format = "wheel";

  src = fetchPypi {
    inherit format pname version;
    dist = abi;
    python = abi;
    inherit abi;
    platform = "manylinux_2_31_x86_64";
    hash = "sha256-vyMsa9hUVwlGJi3269OB5fcJqRdSxNZDNXj0I1lfz1w=";
  };

  pythonImportsCheck = [ "libtpu" ];
  doInstallCheck = false; # requires actual /dev/accel

  meta = with lib; {
    description = "TPU runtime shared library, dlopened by torch_tpu and jaxlib";
    homepage = "https://cloud.google.com/tpu";
    license = licenses.unfree; # "Google Cloud Platform Terms of Service"
    platforms = [ "x86_64-linux" ];
  };
}
