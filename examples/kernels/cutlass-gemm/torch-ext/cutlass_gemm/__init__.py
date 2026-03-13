import torch

from ._ops import ops

def cutlass_gemm(out: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> None:
    ops.cutlass_gemm(out, A, B)
    return out
