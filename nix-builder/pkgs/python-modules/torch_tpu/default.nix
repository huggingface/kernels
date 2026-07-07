{ lib
, buildPythonPackage
, fetchurl
, python312

# jax is not a hard dependency of the torch_tpu wheel, but TPU kernels
# import torch_tpu._internal.pallas (jax_op), which needs jax at import
# time, so it is listed here. jaxlib is not a torch_tpu dependency; it
# is pulled in transitively by jax (nixpkgs' jax declares it).
# libtpu is pinned in-tree via pkgs/python-modules/libtpu.
, jax
, libtpu
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

buildPythonPackage rec {
  pname = "torch_tpu";
  version = "0.1.1.dev20260707090224";
  format = "wheel";

  src = fetchurl {
    url = "https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/torch-tpu/torch_tpu-${version}-cp312-cp312-manylinux_2_31_x86_64.whl";
    hash = "sha256-aTgLY6w1Q6zZvGlBHppNmmUFVthtzD0zoV2nAUoOtBM=";
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

  python = python312; # only cp312 wheels
  dependencies = [ libtpu jax ];

  pythonImportsCheck = [ "torch_tpu" ];
  doInstallCheck = false; # requires actual /dev/accel

  meta = with lib; {
    description = "Torch TPU backend (PrivateUse1 name: \"tpu\")";
    homepage = "https://github.com/google-pytorch/torch_tpu";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
