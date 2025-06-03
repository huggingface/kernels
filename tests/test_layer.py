from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from kernels import (
    Device,
    LayerRepository,
    kernelize,
    register_kernel_mapping,
    use_kernel_forward_from_hub,
)
from kernels.layer import _KERNEL_MAPPING, _validate_layer, use_kernel_mapping

kernel_layer_mapping = {
    "SiluAndMul": {
        Device(type="cuda"): LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
        )
    },
    "SiluAndMulNoCompile": {
        "cuda": LayerRepository(
            repo_id="kernels-test/op-without-fake-test",
            layer_name="SiluAndMul",
        )
    },
    "SiluAndMulStringDevice": {
        "cuda": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
        )
    },
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


@use_kernel_forward_from_hub("SiluAndMulNoCompile")
class SiluAndMulNoCompileKernel(SiluAndMul):
    pass


@use_kernel_forward_from_hub("SiluAndMul")
class SiluAndMulWithKernel(SiluAndMul):
    pass


@use_kernel_forward_from_hub("SiluAndMulStringDevice")
class SiluAndMulStringDevice(SiluAndMul):
    pass


def test_arg_kinds():
    @use_kernel_forward_from_hub("ArgKind")
    class ArgKind(nn.Module):
        def forward(
            self,
            arg1,
            arg2,
            *,
            kwarg1,
            kwarg2=42,
        ):
            return (arg1, arg2, kwarg1, kwarg2)

    arg_kind = ArgKind()
    assert arg_kind("foo", "bar", kwarg1="baz") == ("foo", "bar", "baz", 42)
    assert arg_kind("foo", "bar", kwarg1="baz", kwarg2=5) == ("foo", "bar", "baz", 5)


@pytest.mark.linux_only
@pytest.mark.parametrize("cls", [SiluAndMulWithKernel, SiluAndMulStringDevice])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_hub_forward(cls, device):
    torch.random.manual_seed(0)

    silu_and_mul = SiluAndMul()
    X = torch.randn((32, 64), device=device)
    Y = silu_and_mul(X)

    silu_and_mul_with_kernel = kernelize(cls(), device=device)
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
    silu_and_mul = SiluAndMulWithKernelFallback()
    kernelize(silu_and_mul, device="cuda")


@pytest.mark.linux_only
@pytest.mark.parametrize("cls", [SiluAndMulWithKernel, SiluAndMulNoCompileKernel])
@pytest.mark.parametrize("device", ["cuda"])
def test_torch_compile_layer_without_fallback(cls, device):
    silu_and_mul = SiluAndMul()

    X = torch.randn((32, 64), dtype=torch.float32, device=device)
    Y = silu_and_mul(X)

    silu_and_mul_with_kernel = cls()
    silu_and_mul_with_kernel.eval()

    ctx = (
        pytest.raises(ValueError, match="does not fulfill requirements")
        if cls is SiluAndMulNoCompileKernel
        else nullcontext()
    )
    with ctx:
        silu_and_mul_with_kernel = kernelize(
            silu_and_mul_with_kernel,
            device=device,
            needs_torch_compile=True,
            use_fallback=False,
        )
    silu_and_mul_compiled = torch.compile(silu_and_mul_with_kernel, fullgraph=True)

    Y_compiled = silu_and_mul_compiled(X)

    torch.testing.assert_close(Y_compiled, Y)


@pytest.mark.linux_only
@pytest.mark.parametrize("cls", [SiluAndMulWithKernel, SiluAndMulNoCompileKernel])
@pytest.mark.parametrize("device", ["cuda"])
def test_torch_compile_layer_with_fallback(cls, device):
    silu_and_mul = SiluAndMul()

    X = torch.randn((32, 64), dtype=torch.float32, device=device)
    Y = silu_and_mul(X)

    silu_and_mul_with_kernel = cls()
    silu_and_mul_with_kernel.eval()
    silu_and_mul_with_kernel = kernelize(
        silu_and_mul_with_kernel,
        device=device,
        needs_torch_compile=True,
    )
    silu_and_mul_compiled = torch.compile(silu_and_mul_with_kernel, fullgraph=True)

    Y_compiled = silu_and_mul_compiled(X)

    torch.testing.assert_close(Y_compiled, Y)


def test_mapping_contexts():
    assert set(_KERNEL_MAPPING.get().keys()) == {
        "SiluAndMul",
        "SiluAndMulStringDevice",
        "SiluAndMulNoCompile",
    }

    extra_mapping1 = {
        "TestKernel": {
            Device(type="cuda"): LayerRepository(
                repo_id="kernels-community/activation",
                layer_name="SiluAndMul",
                revision="layers",
            )
        }
    }

    with use_kernel_mapping(extra_mapping1):
        assert set(_KERNEL_MAPPING.get().keys()) == {
            "SiluAndMul",
            "SiluAndMulStringDevice",
            "SiluAndMulNoCompile",
            "TestKernel",
        }

        extra_mapping2 = {
            "SiluAndMul": {
                Device(type="cuda"): LayerRepository(
                    repo_id="kernels-community/non-existing",
                    layer_name="SiluAndMul",
                    revision="layers",
                )
            }
        }

        with use_kernel_mapping(extra_mapping2):
            assert set(_KERNEL_MAPPING.get().keys()) == {
                "SiluAndMul",
                "SiluAndMulStringDevice",
                "SiluAndMulNoCompile",
                "TestKernel",
            }
            assert (
                _KERNEL_MAPPING.get()["SiluAndMul"][Device(type="cuda")].repo_id
                == "kernels-community/non-existing"
            )

        assert set(_KERNEL_MAPPING.get().keys()) == {
            "SiluAndMul",
            "SiluAndMulStringDevice",
            "SiluAndMulNoCompile",
            "TestKernel",
        }
        assert (
            _KERNEL_MAPPING.get()["SiluAndMul"][Device(type="cuda")].repo_id
            == "kernels-community/activation"
        )

        with use_kernel_mapping(extra_mapping2, inherit_mapping=False):
            assert set(_KERNEL_MAPPING.get().keys()) == {
                "SiluAndMul",
            }
            assert (
                _KERNEL_MAPPING.get()["SiluAndMul"][Device(type="cuda")].repo_id
                == "kernels-community/non-existing"
            )

        assert set(_KERNEL_MAPPING.get().keys()) == {
            "SiluAndMul",
            "SiluAndMulStringDevice",
            "SiluAndMulNoCompile",
            "TestKernel",
        }
        assert (
            _KERNEL_MAPPING.get()["SiluAndMul"][Device(type="cuda")].repo_id
            == "kernels-community/activation"
        )

    assert set(_KERNEL_MAPPING.get().keys()) == {
        "SiluAndMul",
        "SiluAndMulStringDevice",
        "SiluAndMulNoCompile",
    }


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


@pytest.mark.linux_only
def test_fallback_used_when_training():
    @use_kernel_forward_from_hub("Linear")
    class TorchLinear(nn.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Used to check that we called hub kernel.
            self.n_calls = 0

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            self.n_calls += 1
            return super().forward(input)

    linear = TorchLinear(32, 32).to("cuda")

    # Case 1: kernel with explicit backward support should always
    #         use the kernel.
    with use_kernel_mapping(
        {
            "Linear": {
                Device(type="cuda"): LayerRepository(
                    repo_id="kernels-test/backward-marker-test",
                    layer_name="LinearBackward",
                )
            }
        }
    ):
        linear.train()
        kernelize(linear)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        assert linear.n_calls == 0

        linear.eval()
        linear(X)
        assert linear.n_calls == 0

    # Case 2: kernel with implicit backward support should always
    #         use the kernel.
    with use_kernel_mapping(
        {
            "Linear": {
                Device(type="cuda"): LayerRepository(
                    repo_id="kernels-test/backward-marker-test",
                    layer_name="LinearImplicitBackward",
                )
            }
        }
    ):
        linear.train()
        kernelize(linear)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        assert linear.n_calls == 0

        linear.eval()
        linear(X)
        assert linear.n_calls == 0

    # Case 3: kernel out backward support should use the kernel in
    #         eval mode and the fallback in training. Test train ->
    #         eval -> train.
    with use_kernel_mapping(
        {
            "Linear": {
                Device(type="cuda"): LayerRepository(
                    repo_id="kernels-test/backward-marker-test",
                    layer_name="LinearNoBackward",
                )
            }
        }
    ):
        linear.train()
        kernelize(linear)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        assert linear.n_calls == 1

        # When switching the kernel to eval, forward gets replaced by
        # the kernel.
        linear.eval()
        linear(X)
        assert linear.n_calls == 1

        ## Let's do it in the other direction to make sure it works as well.
        linear.train()
        linear(X)
        assert linear.n_calls == 2

    # Case 4: same as case 3, but test eval -> train -> eval.
    with use_kernel_mapping(
        {
            "Linear": {
                Device(type="cuda"): LayerRepository(
                    repo_id="kernels-test/backward-marker-test",
                    layer_name="LinearNoBackward",
                )
            }
        }
    ):
        linear.eval()
        kernelize(linear)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        assert linear.n_calls == 2

        linear.train()
        linear(X)
        assert linear.n_calls == 3

        linear.eval()
        linear(X)
        assert linear.n_calls == 3
