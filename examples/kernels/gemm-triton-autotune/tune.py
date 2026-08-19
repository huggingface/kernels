#!/usr/bin/env python3
"""Autotune the GEMM kernel and save the configurations to the source tree.

The tuned configurations are written to
``torch-ext/gemm_triton_autotune/configs/``, named after the GEMM shape and
the current accelerator. Commit these files so that they ship with the
kernel.

Tune the kernel as published on the Hub:

    python tune.py --n 4096 --k 4096

Or tune a local build (see the "Develop locally" chapter of the
kernel-builder documentation for building into ``build/``):

    LOCAL_KERNELS=kernels-test/gemm-triton-autotune=build python tune.py --n 4096 --k 4096
"""

import argparse
import logging
from pathlib import Path

import torch

import kernels

SAVE_DIR = Path(__file__).parent / "torch-ext" / "gemm_triton_autotune" / "configs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True, help="N dimension of the GEMM")
    parser.add_argument("--k", type=int, required=True, help="K dimension of the GEMM")
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=None,
        help="M values to tune (default: a logarithmic range)",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=SAVE_DIR,
        help="Directory to write the configuration file to",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    gemm_kernel = kernels.get_kernel("kernels-test/gemm-triton-autotune", version=1)

    tune_kwargs = {}
    if args.m is not None:
        tune_kwargs["Ms"] = tuple(args.m)

    path = gemm_kernel.tune_gemm(
        N=args.n,
        K=args.k,
        dtype=getattr(torch, args.dtype),
        save_dir=args.save_dir,
        **tune_kwargs,
    )
    print(f"Configurations written to {path}")


if __name__ == "__main__":
    main()
