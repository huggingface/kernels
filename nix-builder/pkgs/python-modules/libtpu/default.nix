{
  lib,
  buildPythonPackage,
  requireFile,
  python,
}:

# libtpu is served from Google's gated Artifact Registry, which requires
# an OAuth2 bearer token. That token can't be handed to a build sandbox
# without making the derivation impure, so instead the wheel is
# prefetched out-of-band and handed to `requireFile`, which only checks
# that a file matching `name` + `hash` already exists in the store:
#
#   GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token) \
#     nix-builder/scripts/helpers/prefetch-tpu-wheel.sh libtpu 0.0.43
#
# The script prints the `hash` line to paste in below. To bump the
# version, update `version`, then re-run the script with the new
# version.
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
  # for cp313 (the nixpkgs default python); if either moves, recompute
  # via the prefetch script above.
  abi = "cp${lib.versions.major python.version}${lib.versions.minor python.version}";
in
buildPythonPackage rec {
  pname = "libtpu";
  version = "0.0.43";
  format = "wheel";

  src = requireFile {
    name = "libtpu-${version}-${abi}-${abi}-manylinux_2_31_x86_64.whl";
    hash = "sha256-X5LVmwJuRcNkMG+m7kKgcaOSKnWjeuw4HtCU+lfWamY="; # cp313
    message = ''
      libtpu is served from a gated Google Artifact Registry and cannot
      be fetched by a pure Nix build. Fetch and register it with:

        GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token) \
          nix-builder/scripts/helpers/prefetch-tpu-wheel.sh libtpu ${version} ${abi}
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
