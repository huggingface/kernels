{
  lib,
  buildPythonPackage,
  fetchurl,
  python,
}:

# Fetched directly from Google's Artifact Registry, which requires an
# OAuth2 bearer token. The token is passed at fetch time through the
# GCLOUD_ACCESS_TOKEN environment variable and turned into a netrc
# entry, so it never appears in the URL or the .drv:
#
#   export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
#
# (With a multi-user Nix daemon the variable must be visible to the
# daemon, not the client shell.)
#
# libtpu ships the `libtpu/libtpu.so` runtime that both torch_tpu and
# jaxlib dlopen. Pinned to 0.0.x to match the jax 0.10.x / torch_tpu
# 0.1.x stack (torch_tpu requires libtpu>=0.0.40; jax's "tpu" extra
# pins libtpu==0.0.43.*), so bump it together with jaxlib/torch_tpu.
#
# NOTE on license: libtpu's wheel METADATA declares its license as
# "Google Cloud Platform Terms of Service" — i.e. unfree, NOT
# Apache-2.0 (unlike torch_tpu). The tpu buildSet therefore sets
# allowUnfree = true (see lib/mk-build-set.nix).

let
  # The wheel ships per-CPython-ABI builds (cp311..cp314); pick the tag
  # matching the python this package set is built for. The hash below is
  # for cp313 (the nixpkgs default python); re-run
  # scripts/helpers/get_torch_tpu_hash.sh when either moves.
  abi = "cp${lib.versions.major python.version}${lib.versions.minor python.version}";
in
buildPythonPackage rec {
  pname = "libtpu";
  version = "0.0.43";
  format = "wheel";

  src = fetchurl {
    url = "https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/libtpu/libtpu-${version}-${abi}-${abi}-manylinux_2_31_x86_64.whl";
    hash = "sha256-X5LVmwJuRcNkMG+m7kKgcaOSKnWjeuw4HtCU+lfWamY="; # cp313
    netrcImpureEnvVars = [ "GCLOUD_ACCESS_TOKEN" ];
    netrcPhase = ''
      if [ -z "''${GCLOUD_ACCESS_TOKEN:-}" ]; then
        echo "GCLOUD_ACCESS_TOKEN is not set; cannot fetch libtpu." >&2
        echo "Run: export GCLOUD_ACCESS_TOKEN=\$(gcloud auth print-access-token)" >&2
        exit 1
      fi
      printf 'machine us-python.pkg.dev\nlogin oauth2accesstoken\npassword %s\n' \
        "$GCLOUD_ACCESS_TOKEN" > netrc
    '';
  };

  dependencies = [ ];

  pythonImportsCheck = [ "libtpu" ];
  doInstallCheck = false; # requires actual /dev/accel

  meta = with lib; {
    description = "TPU runtime shared library, dlopened by torch_tpu and jaxlib";
    homepage = "https://cloud.google.com/tpu";
    license = licenses.unfree; # "Google Cloud Platform Terms of Service"
    platforms = [ "x86_64-linux" ];
  };
}
