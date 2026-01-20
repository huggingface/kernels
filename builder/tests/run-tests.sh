#!/bin/bash

PYTHONPATH="relu-kernel:cutlass-gemm-kernel:$PYTHONPATH" \
  .venv/bin/pytest relu_tests cutlass_gemm_tests

# We only care about importing, the kernel is trivial.
PYTHONPATH="silu-and-mul-kernel:$PYTHONPATH" \
  .venv/bin/python -c "import silu_and_mul"

PYTHONPATH="relu-kernel-cpu:$PYTHONPATH" \
   CUDA_VISIBLE_DEVICES="" \
  .venv/bin/pytest relu_tests
