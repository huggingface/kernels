import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kernels import (
    FuncRepository,
    LayerRepository,
    LocalFuncRepository,
    Mode,
    install_kernel,
    kernelize,
    use_kernel_func_from_hub,
    use_kernel_mapping,
)


# A function + layer that we can map arbitrary functions to for testing.
@use_kernel_func_from_hub("surprise_me")
def surprise_me(x: torch.Tensor):
    return x


class SurpriseMe(nn.Module):
    def __init__(self):
        super().__init__()
        self.surprise_me = surprise_me

    def forward(self, x: torch.Tensor):
        return self.surprise_me(x)


def test_decorator():
    @use_kernel_func_from_hub("identity_func")
    def identity(x):
        return x

    assert type(identity).kernel_layer_name == "identity_func"
    assert isinstance(identity, nn.Module)


def test_kernel_func(device):
    model = SurpriseMe()

    x = torch.arange(-10, 10, device=device).float()
    assert model(x) is x

    with use_kernel_mapping(
        {
            "surprise_me": {
                device: FuncRepository(
                    "kernels-test/flattened-build",
                    func_name="silu_and_mul",
                )
            }
        }
    ):
        model = kernelize(model, mode=Mode.INFERENCE, device=device)

    torch.testing.assert_close(model(x), _silu_and_mul(x))

    # And empty mapping should revert to the original implementation.
    with use_kernel_mapping({"surprise_me": {}}):
        model = kernelize(model, mode=Mode.INFERENCE, device=device)

    assert model(x) is x


@pytest.mark.cuda_only
def test_kernel_func_with_layer():
    model = SurpriseMe()

    x = torch.arange(-10, 10, device="cuda").float()
    assert model(x) is x

    # We can also replace the function by a pure layer.
    with use_kernel_mapping(
        {
            "surprise_me": {
                "cuda": LayerRepository(
                    "kernels-community/activation",
                    layer_name="SiluAndMul",
                )
            }
        }
    ):
        model = kernelize(model, mode=Mode.INFERENCE, device="cuda")

    torch.testing.assert_close(model(x), _silu_and_mul(x))

    # And empty mapping should revert to the original implementation.
    with use_kernel_mapping({"surprise_me": {}}):
        model = kernelize(model, mode=Mode.INFERENCE, device="cuda")

    assert model(x) is x


def test_local_kernel_func(device):
    model = SurpriseMe()

    x = torch.arange(-10, 10).float()
    assert model(x) is x

    package_name, path = install_kernel("kernels-test/flattened-build", "main")

    with use_kernel_mapping(
        {
            "surprise_me": {
                device: LocalFuncRepository(
                    repo_path=path.parent.parent,
                    package_name=package_name,
                    func_name="silu_and_mul",
                )
            }
        }
    ):
        model = kernelize(model, mode=Mode.INFERENCE, device=device)

    torch.testing.assert_close(model(x), _silu_and_mul(x))

    with use_kernel_mapping({"do_something_func": {}}):
        model = kernelize(model, mode=Mode.INFERENCE, device=device)

    assert model(x) is x


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]
