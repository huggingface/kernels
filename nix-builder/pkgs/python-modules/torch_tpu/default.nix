{
  lib,
  autoPatchelfHook,
  buildPythonPackage,
  fetchurl,
  python,
  stdenv,

  # jax is not a hard dependency of the torch_tpu wheel, but TPU kernels
  # import torch_tpu._internal.pallas (jax_op), which needs jax at import
  # time, so it is listed here. jaxlib is not a torch_tpu dependency; it
  # is pulled in transitively by jax (nixpkgs' jax declares it).
  # libtpu is pinned in-tree via pkgs/python-modules/libtpu.
  jax,
  libtpu,

  # The remaining wheel Requires-Dist entries (numpy and absl-py come in
  # transitively through jax).
  frozendict,
  immutabledict,
  portpicker,
  tensorboard,
  torch,
}:

# Fetched directly from Google's Artifact Registry, which requires an
# OAuth2 bearer token. The token is passed at fetch time through the
# GCLOUD_ACCESS_TOKEN environment variable and turned into a netrc
# entry, so it never appears in the URL or the .drv:
#
#   export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
#
# (With a multi-user Nix daemon the variable must be visible to the
# daemon, not the client shell.) The dev build is dated; bump the date
# suffix in `version` and re-run
# scripts/helpers/get_torch_tpu_hash.sh when refreshing.

let
  # The wheel ships per-CPython-ABI builds (cp311..cp314); pick the tag
  # matching the python this package set is built for. The hash below is
  # for cp313 (the nixpkgs default python); re-run
  # scripts/helpers/get_torch_tpu_hash.sh when either moves.
  abi = "cp${lib.versions.major python.version}${lib.versions.minor python.version}";
in
buildPythonPackage rec {
  pname = "torch_tpu";
  version = "0.1.1.dev20260707090224";
  format = "wheel";

  src = fetchurl {
    url = "https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/torch-tpu/torch_tpu-${version}-${abi}-${abi}-manylinux_2_31_x86_64.whl";
    hash = "sha256-eCjwoX0UKd/L/IccxXn88p0GdyQjFdPl2Er1luZz4H0="; # cp313
    netrcImpureEnvVars = [ "GCLOUD_ACCESS_TOKEN" ];
    netrcPhase = ''
      if [ -z "''${GCLOUD_ACCESS_TOKEN:-}" ]; then
        echo "GCLOUD_ACCESS_TOKEN is not set; cannot fetch torch_tpu." >&2
        echo "Run: export GCLOUD_ACCESS_TOKEN=\$(gcloud auth print-access-token)" >&2
        exit 1
      fi
      printf 'machine us-python.pkg.dev\nlogin oauth2accesstoken\npassword %s\n' \
        "$GCLOUD_ACCESS_TOKEN" > netrc
    '';
  };

  dependencies = [
    libtpu
    jax
    frozendict
    immutabledict
    portpicker
    tensorboard
    torch
  ];

  # The bundled extensions link against libtorch_python.so / libc10.so /
  # libtorch_cpu.so, which live in torch's site-packages, not on the
  # default search path.
  nativeBuildInputs = [ autoPatchelfHook ];
  buildInputs = [ stdenv.cc.cc.lib ];
  preFixup = ''
    addAutoPatchelfSearchPath ${torch}/${python.sitePackages}/torch/lib
  '';

  pythonImportsCheck = [ "torch_tpu" ];
  doInstallCheck = false; # requires actual /dev/accel

  meta = with lib; {
    description = "Torch TPU backend (PrivateUse1 name: \"tpu\")";
    homepage = "https://github.com/google-pytorch/torch_tpu";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
