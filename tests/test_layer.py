import re

import torch
import torch.nn as nn
from torch.nn import functional as F

from kernels import use_hub_kernel


class SiluAndMul(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        d = input.shape[-1] // 2
        return F.silu(input[..., :d]) * input[..., d:]


def test_activation_layer():
    @use_hub_kernel(
        "kernels-community/activation",
        layer_name="SiluAndMul",
        revision="layers",
        fallback_on_error=False,
    )
    class SiluAndMulWithKernel(SiluAndMul):
        pass

    torch.random.manual_seed(0)

    silu_and_mul = SiluAndMul()
    X = torch.randn((32, 64), device="cuda")
    Y = silu_and_mul(X)

    # Verify that the Hub kernel was loaded.
    assert SiluAndMulWithKernel.__name__ == "SiluAndMul"
    assert re.match(r"activation.*layers", SiluAndMulWithKernel.__module__)

    silu_and_mul_with_kernel = SiluAndMulWithKernel()
    Y_kernel = silu_and_mul_with_kernel(X)

    torch.testing.assert_close(Y_kernel, Y)


def test_layer_fallback_works():
    @use_hub_kernel("kernels-community/non-existing", layer_name="SiluAndMul")
    class SiluAndMulWithKernelFallback(SiluAndMul):
        pass

    # Check that we don't raise an exception for a non-existing kernel.
    SiluAndMulWithKernelFallback()
