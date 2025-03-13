import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from kernels import (
    Device,
    LayerRepository,
    register_kernel_mapping,
    use_kernel_forward_from_hub,
)
from kernels.layer import _validate_layer

kernel_layer_mapping = {
    "SiluAndMul": {
        Device(type="cuda"): LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
            revision="layers",
        )
    }
}

register_kernel_mapping(kernel_layer_mapping)


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()
        # Used to check that we called hub kernel.
        self.n_calls = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.n_calls += 1
        d = input.shape[-1] // 2
        return F.silu(input[..., :d]) * input[..., d:]


@use_kernel_forward_from_hub("SiluAndMul")
class SiluAndMulWithKernel(SiluAndMul):
    pass


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_hub_forward(device):
    torch.random.manual_seed(0)

    silu_and_mul = SiluAndMul()
    X = torch.randn((32, 64), device=device)
    Y = silu_and_mul(X)

    silu_and_mul_with_kernel = SiluAndMulWithKernel()
    Y_kernel = silu_and_mul_with_kernel(X)

    torch.testing.assert_close(Y_kernel, Y)

    assert silu_and_mul.n_calls == 1
    if device == "cuda":
        assert silu_and_mul_with_kernel.n_calls == 0
    else:
        assert silu_and_mul_with_kernel.n_calls == 1


def test_layer_fallback_works():
    @use_kernel_forward_from_hub("SiluAndMulNonExisting")
    class SiluAndMulWithKernelFallback(SiluAndMul):
        pass

    # Check that we don't raise an exception for a non-existing kernel.
    SiluAndMulWithKernelFallback()


def test_validate_kernel_layer():
    class BadLayer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.foo = 42

    with pytest.raises(TypeError, match="not override"):
        _validate_layer(cls=BadLayer, check_cls=SiluAndMul)

    class BadLayer2(nn.Module):
        foo: int = 42

    with pytest.raises(TypeError, match="not contain additional members"):
        _validate_layer(cls=BadLayer2, check_cls=SiluAndMul)

    class BadLayer3(nn.Module):
        def forward(self, x: torch.Tensor, foo: int) -> torch.Tensor: ...

    with pytest.raises(TypeError, match="different number of arguments"):
        _validate_layer(cls=BadLayer3, check_cls=SiluAndMul)

    class BadLayer4(nn.Module):
        def forward(self, *, x: torch.Tensor) -> torch.Tensor: ...

    with pytest.raises(TypeError, match="different kind of arguments"):
        _validate_layer(cls=BadLayer4, check_cls=SiluAndMul)
