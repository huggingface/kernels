import torch

from cutlass_gemm import cutlass_gemm

def test_gemm():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        A = torch.randn((64, 32), device=torch.device("xpu"), dtype=torch.bfloat16)
        B = torch.randn((32, 64), device=torch.device("xpu"), dtype=torch.bfloat16)
        out = torch.randn((64, 64), device=torch.device("xpu"), dtype=torch.float32)
    else:
        A = torch.randn((10, 20), device=torch.device("cuda"), dtype=torch.float32)
        B = torch.randn((20, 30), device=torch.device("cuda"), dtype=torch.float32)
        out = torch.randn((10, 30), device=torch.device("cuda"), dtype=torch.float32)

    cutlass_gemm(out, A, B)

    torch.testing.assert_allclose(out, torch.mm(A.float(), B.float()))
