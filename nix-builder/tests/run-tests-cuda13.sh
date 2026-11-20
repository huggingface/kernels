#!/bin/bash
set -euo pipefail

RELU_RUST_CUDA_OXIDE_PATH=$(echo relu-rust-cuda-oxide-kernel/tvm-ffi*)
RELU_RUST_CUTILE_PATH=$(echo relu-rust-cutile-kernel/tvm-ffi*)

CUTILE_TILEIRAS_PATH=$(find .venv -path '*/nvidia/cu13/bin/tileiras' -print -quit)
if [ -z "${CUTILE_TILEIRAS_PATH}" ]; then
  echo "Cannot find tileiras from nvidia-cuda-tileiras" >&2
  exit 1
fi
export CUTILE_TILEIRAS_PATH

LOCAL_KERNELS="kernels-test/relu-rust-cuda-oxide=${RELU_RUST_CUDA_OXIDE_PATH}:kernels-test/relu-rust-cutile=${RELU_RUST_CUTILE_PATH}" \
  .venv/bin/pytest relu_rust_cuda_oxide_tests relu_rust_cutile_tests
