#!/bin/bash
set -euo pipefail

# Expand to build variant directories.
EXTRA_DATA_PATH=$(echo extra-data/torch*)
RELU_PATH=$(echo relu-kernel/torch*)
RELU_TORCH_STABLE_ABI_PATH=$(echo relu-torch-stable-abi-kernel/torch*)
RELU_TVM_FFI_PATH=$(echo relu-tvm-ffi-kernel/tvm-ffi*)
RELU_CUDA_OXIDE_PATH=$(echo relu-cuda-oxide-kernel/tvm-ffi*)
CUTLASS_PATH=$(echo cutlass-gemm-kernel/torch*)
CUTLASS_TVM_FFI_PATH=$(echo cutlass-gemm-tvm-ffi-kernel/tvm-ffi*)
RELU_TRITON_PATH=$(echo relu-triton-kernel/torch*)
SILU_MUL_PATH=$(echo silu-and-mul-kernel/torch*)
RELU_CPU_PATH=$(echo relu-kernel-cpu/torch*)
CPP20_SYMBOLS_PATH=$(echo cpp20-symbols-kernel/torch*)

LOCAL_KERNELS="kernels-test/extra-data=${EXTRA_DATA_PATH}:kernels-test/relu=${RELU_PATH}:kernels-test/relu-torch-stable-abi=${RELU_TORCH_STABLE_ABI_PATH}:kernels-test/relu-tvm-ffi=${RELU_TVM_FFI_PATH}:kernels-test/relu-cuda-oxide=${RELU_CUDA_OXIDE_PATH}:kernels-test/cutlass-gemm=${CUTLASS_PATH}:kernels-test/cutlass-gemm-tvm-ffi=${CUTLASS_TVM_FFI_PATH}" \
  .venv/bin/pytest extra_data_tests relu_tests relu_tvm_ffi_tests relu_cuda_oxide_tests cutlass_gemm_tests cutlass_gemm_tvm_ffi_tests

LOCAL_KERNELS="kernels-test/relu-triton=${RELU_TRITON_PATH}" \
  .venv/bin/pytest relu_triton_tests

# We only care about importing, the kernel is trivial.
LOCAL_KERNELS="kernels-test/silu-and-mul=${SILU_MUL_PATH}" \
  .venv/bin/python -c "import kernels; kernels.get_kernel('kernels-test/silu-and-mul', version=1)"

LOCAL_KERNELS="kernels-test/relu=${RELU_CPU_PATH}" \
   CUDA_VISIBLE_DEVICES="" \
  .venv/bin/pytest relu_tests

LOCAL_KERNELS="kernels-test/cpp20-symbols=${CPP20_SYMBOLS_PATH}" \
   CUDA_VISIBLE_DEVICES="" \
  .venv/bin/pytest cpp20_symbols_tests
