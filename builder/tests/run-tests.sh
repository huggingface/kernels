#!/bin/bash

# Expand to build variant directories.
EXTRA_DATA_PATH=$(echo extra-data/torch*)
RELU_PATH=$(echo relu-kernel/torch*)
CUTLASS_PATH=$(echo cutlass-gemm-kernel/torch*)
SILU_MUL_PATH=$(echo silu-and-mul-kernel/torch*)
RELU_CPU_PATH=$(echo relu-kernel-cpu/torch*)

PYTHONPATH="$EXTRA_DATA_PATH:$RELU_PATH:$CUTLASS_PATH:$PYTHONPATH" \
  .venv/bin/pytest extra_data_tests relu_tests cutlass_gemm_tests

# We only care about importing, the kernel is trivial.
PYTHONPATH="$SILU_MUL_PATH:$PYTHONPATH" \
  .venv/bin/python -c "import silu_and_mul"

PYTHONPATH="$RELU_CPU_PATH:$PYTHONPATH" \
   CUDA_VISIBLE_DEVICES="" \
  .venv/bin/pytest relu_tests
