import copy
import logging
import pickle
from pathlib import Path
from types import MethodType

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
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub,
    use_kernel_mapping,
    use_kernelized_func,
)
from kernels.layer.func import LockedFuncRepository


# Base modules used as a replacement forward in tests
class AddOne(nn.Module):
    def forward(self, x):
        return x + 1


class TimesThree(nn.Module):
    def forward(self, x):
        return x * 3


# Functions + layers used to test function kernelization
@use_kernel_forward_from_hub("surprise_me")
def surprise_me(x: torch.Tensor):
    return x


@use_kernelized_func(surprise_me)
class SurpriseMe(nn.Module):
    def forward(self, x: torch.Tensor):
        return surprise_me(x)


@use_kernel_forward_from_hub("double_me")
def double_me(x: torch.Tensor):
    return x * 2


@use_kernelized_func(double_me)
class Inner(nn.Module):
    def forward(self, x):
        return double_me(x)


# To check nested modules
@use_kernelized_func(surprise_me)
class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()

    def forward(self, x):
        # The second call also verifies that Inner restored the outer context.
        return surprise_me(self.inner(surprise_me(x)))


def test_decorator():
    @use_kernel_forward_from_hub("identity_func")
    def identity(x):
        return x

    assert type(identity).kernel_layer_name == "identity_func"
    assert isinstance(identity, nn.Module)


def test_deprecated_decorator():
    @use_kernel_func_from_hub("identity_func")
    def identity(x):
        return x

    assert type(identity).kernel_layer_name == "identity_func"
    assert isinstance(identity, nn.Module)


def test_deprecated_func_repository_requires_version_or_revision():
    with pytest.raises(ValueError, match="Either a revision or a version must be specified"):
        FuncRepository("kernels-test/flattened-build", func_name="silu_and_mul")


def test_deprecated_func_repository(device):
    model = SurpriseMe()

    x = torch.arange(-10, 10, device=device).float()
    assert model(x) is x

    with use_kernel_mapping(
        {
            "surprise_me": {
                device: FuncRepository(
                    "kernels-test/flattened-build",
                    func_name="silu_and_mul",
                    revision="main",
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
                    "kernels-test/silu-and-mul",
                    layer_name="SiluAndMul",
                    version=1,
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


def test_deprecated_local_kernel_func(device):
    model = SurpriseMe()

    x = torch.arange(-10, 10).float()
    assert model(x) is x

    path = install_kernel("kernels-test/flattened-build", revision="main")

    with use_kernel_mapping(
        {
            "surprise_me": {
                device: LocalFuncRepository(
                    repo_path=path.parent.parent,
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


def test_deprecated_kernel_func(caplog):
    with caplog.at_level(logging.WARNING, logger="kernels.layer.func"):
        FuncRepository("kernels-test/flattened-build", func_name="silu_and_mul", version=1)

        project_dir = Path(__file__).parent / "layer_locking"
        LockedFuncRepository(
            "kernels-test/versions",
            func_name="version",
            lockfile=project_dir / "kernels.lock",
        )

        LocalFuncRepository(
            # We are never loading the kernel, so we can just use an invalid path.
            repo_path=Path("."),
            func_name="silu_and_mul",
        )

        @use_kernel_func_from_hub("deprecated")
        def deprecated_func(x):
            return x

    assert caplog.text.count("kernels 0.17") == 4


def test_use_kernelized_func_used_on_non_kernelized_func():
    def not_kernelized(x):
        return x

    with pytest.raises(ValueError, match="Function `not_kernelized` is not decorated"):

        @use_kernelized_func(not_kernelized)
        class NotKernelized(nn.Module):
            def forward(self, x: torch.Tensor):
                return not_kernelized(x)


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


# Imitates the kernels exchange
def _bind_forward(wrapper, forward):
    wrapper.forward = MethodType(forward, wrapper)


def test_kernel_func_is_per_instance():
    a, b = SurpriseMe(), SurpriseMe()

    a_func = a._kernel_funcs["surprise_me"]
    b_func = b._kernel_funcs["surprise_me"]

    assert a_func is not b_func
    assert a_func is not surprise_me
    assert b_func is not surprise_me

    _bind_forward(a_func, AddOne.forward)

    x = torch.arange(4).float()
    torch.testing.assert_close(a(x), x + 1)
    torch.testing.assert_close(b(x), x)


@pytest.mark.parametrize("copy_fn", [copy.copy, copy.deepcopy])
def test_kernel_func_copy_is_independent(copy_fn):
    model = SurpriseMe()
    kernel_func = model._kernel_funcs["surprise_me"]
    _bind_forward(kernel_func, AddOne.forward)

    copied = copy_fn(kernel_func)

    assert copied is not kernel_func
    assert copied is not surprise_me
    assert copied.__dict__["forward"].__self__ is copied

    x = torch.arange(4).float()
    torch.testing.assert_close(copied(x), x + 1)
    torch.testing.assert_close(kernel_func(x), x + 1)


@pytest.mark.parametrize(
    "restore_fn",
    [
        copy.deepcopy,
        lambda model: pickle.loads(pickle.dumps(model)),
    ],
)
def test_kernel_func_serialization_is_independent(restore_fn):
    model = SurpriseMe()
    _bind_forward(model._kernel_funcs["surprise_me"], AddOne.forward)

    restored = restore_fn(model)

    assert restored._kernel_funcs["surprise_me"] is not model._kernel_funcs["surprise_me"]
    assert restored._kernel_funcs["surprise_me"] is not surprise_me
    assert restored._kernel_funcs["surprise_me"].__dict__["forward"].__self__ is restored._kernel_funcs["surprise_me"]

    x = torch.arange(4).float()
    torch.testing.assert_close(restored(x), x + 1)

    # Resetting the restored model must not affect the original
    with use_kernel_mapping({"surprise_me": {}}, inherit_mapping=False):
        kernelize(restored, device="cpu", mode=Mode.INFERENCE)

    torch.testing.assert_close(restored(x), x)
    torch.testing.assert_close(model(x), x + 1)


@pytest.mark.parametrize("compile", [False, True])
def test_kernel_func_nested_dispatch(compile):
    model = Outer()

    _bind_forward(model._kernel_funcs["surprise_me"], AddOne.forward)
    _bind_forward(model.inner._kernel_funcs["double_me"], TimesThree.forward)

    if compile:
        # We only need to know whether it's safe around get/set so the backend is not relevant
        model = torch.compile(model, backend="eager", fullgraph=True)

    x = torch.tensor(1.0)

    # Outer (+1) -> Inner (*3) -> Outer (+1): 1 -> 2 -> 6 -> 7
    torch.testing.assert_close(model(x), torch.tensor(7.0))
