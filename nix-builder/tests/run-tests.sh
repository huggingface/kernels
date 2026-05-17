#!/bin/bash
set -euo pipefail

# Expand to build variant directories.
EXTRA_DATA_PATH=$(echo extra-data/torch*)
RELU_PATH=$(echo relu-kernel/torch*)
RELU_TVM_FFI_PATH=$(echo relu-tvm-ffi-kernel/tvm-ffi*)
CUTLASS_PATH=$(echo cutlass-gemm-kernel/torch*)
CUTLASS_TVM_FFI_PATH=$(echo cutlass-gemm-tvm-ffi-kernel/tvm-ffi*)
SILU_MUL_PATH=$(echo silu-and-mul-kernel/torch*)
RELU_CPU_PATH=$(echo relu-kernel-cpu/torch*)
CPP20_SYMBOLS_PATH=$(echo cpp20-symbols-kernel/torch*)

PYTHONPATH="$EXTRA_DATA_PATH:$RELU_PATH:$RELU_TVM_FFI_PATH:$CUTLASS_PATH:$CUTLASS_TVM_FFI_PATH" \
  .venv/bin/pytest extra_data_tests relu_tests relu_tvm_ffi_tests cutlass_gemm_tests cutlass_gemm_tvm_ffi_tests

# We only care about importing, the kernel is trivial.
PYTHONPATH="$SILU_MUL_PATH" \
  .venv/bin/python -c "import silu_and_mul"

PYTHONPATH="$RELU_CPU_PATH" \
   CUDA_VISIBLE_DEVICES="" \
  .venv/bin/pytest relu_tests

PYTHONPATH="$CPP20_SYMBOLS_PATH" \
   CUDA_VISIBLE_DEVICES="" \
  .venv/bin/pytest cpp20_symbols_tests
