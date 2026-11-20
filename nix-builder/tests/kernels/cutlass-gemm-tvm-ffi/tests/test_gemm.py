import kernels
import pytest
import torch

cutlass_gemm_tvm_ffi = kernels.get_kernel(
    "kernels-test/cutlass-gemm-tvm-ffi", version=1
)


@pytest.mark.kernels_ci
def test_gemm(device):
    A = torch.randn((64, 32), device=device, dtype=torch.float32)
    B = torch.randn((32, 64), device=device, dtype=torch.float32)
    out = torch.zeros((64, 64), device=device, dtype=torch.float32)

    cutlass_gemm_tvm_ffi.cutlass_gemm(out, A, B)

    torch.testing.assert_close(out, torch.mm(A, B))
