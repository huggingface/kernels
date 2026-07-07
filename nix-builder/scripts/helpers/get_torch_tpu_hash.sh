#!/usr/bin/env bash
# Print the wheel URL and SRI hash for a torch-tpu-virtual-registry
# package, so a developer can paste the result straight into
# pkgs/python-modules/{torch_tpu,libtpu}/default.nix.
#
# The registry's simple index publishes each wheel's sha256 as a URL
# fragment, so no download is needed to compute the hash.
#
# Usage:
#   bash scripts/helpers/get_torch_tpu_hash.sh torch_tpu 0.1.1.dev20260707090224
#   bash scripts/helpers/get_torch_tpu_hash.sh libtpu    0.0.43
#
# Requires `gcloud auth login` once, plus curl and python3.
set -euo pipefail
pkg="${1:?usage: $0 <pkg> <version>, e.g. $0 torch_tpu 0.1.1.dev20260707090224}"
ver="${2:?usage: $0 <pkg> <version>, e.g. $0 torch_tpu 0.1.1.dev20260707090224}"
token="$(gcloud auth print-access-token)"
index="https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/${pkg//_/-}/"
index_html="$(curl -sSf -u "oauth2accesstoken:${token}" "${index}")"
href="$(grep -o 'href="[^"]*"' <<<"${index_html}" \
    | grep -F "/${pkg}-${ver}-cp312-" \
    | sed 's/^href="//; s/"$//' \
    | head -1 || true)"
if [ -z "${href}" ]; then
    echo "No cp312 wheel for ${pkg}==${ver} in ${index}" >&2
    exit 1
fi
echo "URL:    ${href%%#*}"
echo "Hash:   $(python3 -c \
    "import base64,sys; print('sha256-' + base64.b64encode(bytes.fromhex(sys.argv[1])).decode())" \
    "${href##*#sha256=}")"
