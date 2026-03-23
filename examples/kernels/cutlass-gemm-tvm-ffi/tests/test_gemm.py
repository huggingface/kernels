import torch
from cutlass_gemm_tvm_ffi import cutlass_gemm


def test_gemm(device):
    A = torch.randn((64, 32), device=device, dtype=torch.float32)
    B = torch.randn((32, 64), device=device, dtype=torch.float32)
    out = torch.zeros((64, 64), device=device, dtype=torch.float32)

    cutlass_gemm(out, A, B)

    torch.testing.assert_close(out, torch.mm(A, B))
