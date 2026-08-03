#!/usr/bin/env bash
# Fetch a wheel from Google's gated torch-tpu-virtual-registry and
# register it in the Nix store under the exact (name, sha256) that
# `requireFile` expects in pkgs/python-modules/{libtpu,torch_tpu}/default.nix.
#
# The fetch (and the OAuth2 token it needs) happens here, out-of-band,
# so the actual Nix build stays pure and never touches the network.
#
# Usage:
#   GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token) \
#     scripts/helpers/prefetch-tpu-wheel.sh libtpu 0.0.43 [cp313]
#   GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token) \
#     scripts/helpers/prefetch-tpu-wheel.sh torch_tpu 0.1.1.dev20260707090224 [cp313]
#
# Prints a `hash = "sha256-...";` line to paste into the matching
# default.nix when bumping a version. The third argument is the
# CPython ABI tag (default cp313, matching the nixpkgs default python
# used by the builder).
#
# Requires `gcloud auth login` once, plus nix (with the `nix-command`
# experimental feature) and jq.
set -euo pipefail

pkg="${1:?usage: $0 <libtpu|torch_tpu> <version> [abi]}"
version="${2:?usage: $0 <libtpu|torch_tpu> <version> [abi]}"
abi="${3:-cp313}"
: "${GCLOUD_ACCESS_TOKEN:?set GCLOUD_ACCESS_TOKEN to \$(gcloud auth print-access-token)}"

registry_pkg="${pkg//_/-}"
name="${pkg}-${version}-${abi}-${abi}-manylinux_2_31_x86_64.whl"
url="https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/${registry_pkg}/${name}"
netrc="$(mktemp)"
trap 'rm -f "${netrc}"' EXIT
printf 'machine us-python.pkg.dev\nlogin oauth2accesstoken\npassword %s\n' \
  "${GCLOUD_ACCESS_TOKEN}" >"${netrc}"

hash="$(nix store prefetch-file --json --hash-type sha256 --name "${name}" \
  --option netrc-file "${netrc}" "${url}" | jq -r .hash)"

echo "hash = \"${hash}\"; # ${abi}"
