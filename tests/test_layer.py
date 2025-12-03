import sys
from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from kernels import (
    CUDAProperties,
    Device,
    FuncRepository,
    LayerRepository,
    LocalLayerRepository,
    Mode,
    kernelize,
    register_kernel_mapping,
    use_kernel_forward_from_hub,
    use_kernel_mapping,
)
from kernels.layer.layer import (
    _KERNEL_MAPPING,
    _validate_layer,
)
from kernels.utils import (
    install_kernel,
)

kernel_layer_mapping = {
    "SiluAndMul": {
        Device(type="cuda"): LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
        ),
        "npu": LayerRepository(
            repo_id="kernels-ext-npu/SwiGlu",
            layer_name="SwiGlu",
        ),
    },
    "SiluAndMulNoCompile": {
        "cuda": LayerRepository(
            repo_id="kernels-test/op-without-fake-test",
            layer_name="SiluAndMul",
        ),
        "rocm": LayerRepository(
            repo_id="kernels-test/op-without-fake-test",
            layer_name="SiluAndMul",
        ),
    },
    "SiluAndMulStringDevice": {
        "cuda": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
        )
    },
    "LigerRMSNorm": {
        "xpu": LayerRepository(
            repo_id="kernels-community/liger_kernels",
            layer_name="LigerRMSNorm",  # Triton
        )
    },
}

register_kernel_mapping(kernel_layer_mapping)


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        # Used to check that we called hub kernel.
        self.n_calls = 0
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor):
        self.n_calls += 1
        var = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.variance_epsilon)
        return x_norm * self.weight


@use_kernel_forward_from_hub("LigerRMSNorm")
class RMSNormWithKernel(RMSNorm):
    pass


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


@use_kernel_forward_from_hub("Linear")
class TorchLinearWithCounter(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Used to check that we called hub kernel.
        self.n_calls = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.n_calls += 1
        return super().forward(input)


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


@pytest.mark.cuda_only
@pytest.mark.parametrize("cls", [SiluAndMulWithKernel, SiluAndMulStringDevice])
def test_hub_forward(cls):
    torch.random.manual_seed(0)

    silu_and_mul = SiluAndMul()
    X = torch.randn((32, 64), device="cuda")
    Y = silu_and_mul(X)

    silu_and_mul_with_kernel = kernelize(cls(), device="cuda", mode=Mode.INFERENCE)
    Y_kernel = silu_and_mul_with_kernel(X)

    torch.testing.assert_close(Y_kernel, Y)

    assert silu_and_mul.n_calls == 1
    assert silu_and_mul_with_kernel.n_calls == 0


@pytest.mark.cuda_only
@pytest.mark.parametrize("cls", [SiluAndMulWithKernel, SiluAndMulStringDevice])
def test_hub_func(cls):
    torch.random.manual_seed(0)

    silu_and_mul = SiluAndMul()
    X = torch.randn((32, 64), device="cuda")
    Y = silu_and_mul(X)

    # SiluAndMul is pure, so we can also use a function.
    with use_kernel_mapping(
        {
            "surprise_me": {
                "cuda": FuncRepository(
                    "kernels-test/flattened-build",
                    func_name="silu_and_mul",
                )
            }
        }
    ):
        silu_and_mul_with_kernel = kernelize(cls(), device="cuda", mode=Mode.INFERENCE)
    Y_kernel = silu_and_mul_with_kernel(X)

    torch.testing.assert_close(Y_kernel, Y)

    assert silu_and_mul.n_calls == 1
    assert silu_and_mul_with_kernel.n_calls == 0


@pytest.mark.rocm_only
def test_hub_forward_rocm():
    torch.manual_seed(0)

    silu_and_mul = SiluAndMul()
    X = torch.randn((32, 64))
    Y = silu_and_mul(X)

    silu_and_mul_with_kernel = kernelize(
        SiluAndMulNoCompileKernel(), device="rocm", mode=Mode.INFERENCE
    )
    Y_kernel = silu_and_mul_with_kernel(X)

    torch.testing.assert_close(Y_kernel, Y)

    assert silu_and_mul.n_calls == 1
    # Should use kernel (n_calls == 0) if ROCm kernel is available, otherwise fallback (n_calls == 1)
    # The exact behavior depends on whether the test kernel exists for ROCm
    assert silu_and_mul_with_kernel.n_calls in [0, 1]


@pytest.mark.xpu_only
def test_hub_forward_xpu():
    torch.manual_seed(0)

    hidden_size = 1024
    weight = torch.ones(hidden_size, device="xpu")
    rms_norm = RMSNorm(weight).to("xpu")
    X = torch.randn(4, 16, hidden_size, device="xpu", dtype=torch.float32)
    Y = rms_norm(X)

    rms_norm_with_kernel = kernelize(
        RMSNormWithKernel(weight), mode=Mode.INFERENCE, device="xpu"
    )
    Y_kernel = rms_norm_with_kernel(X)

    torch.testing.assert_close(Y_kernel, Y)

    assert rms_norm.n_calls == 1
    assert rms_norm_with_kernel.n_calls == 0


@pytest.mark.npu_only
def test_hub_forward_npu():
    torch.manual_seed(0)

    silu_and_mul = SiluAndMul()
    X = torch.randn((32, 64), device="npu")
    Y = silu_and_mul(X)

    silu_and_mul_with_kernel = kernelize(
        SiluAndMulWithKernel(), device="npu", mode=Mode.INFERENCE
    )
    Y_kernel = silu_and_mul_with_kernel(X)

    torch.testing.assert_close(Y_kernel, Y)

    assert silu_and_mul.n_calls == 1
    assert silu_and_mul_with_kernel.n_calls == 0


def test_rocm_kernel_mapping(device):
    """Test that ROCm shorthand device mapping works correctly."""

    # Lookup uses the GPU capability, so it fails for non-ROCm/CUDA.
    if device not in ["cuda", "rocm"]:
        pytest.skip("Test only applicable to CUDA and ROCM devices")

    kernel_layer_mapping = {
        "SiluAndMul": {
            "rocm": LayerRepository(
                repo_id="kernels-community/activation",
                layer_name="SiluAndMul",
            )
        }
    }

    # Test that the mapping is processed correctly
    with use_kernel_mapping(kernel_layer_mapping, inherit_mapping=False):
        mapping = _KERNEL_MAPPING.get()

        # Verify the mapping exists
        assert "SiluAndMul" in mapping
        assert "rocm" in mapping["SiluAndMul"]

        # Verify the repository is correctly stored
        rocm_repos = mapping["SiluAndMul"]["rocm"]
        assert rocm_repos is not None
        assert (
            rocm_repos.repos[Mode.FALLBACK]._repo_id == "kernels-community/activation"
        )
        assert rocm_repos.repos[Mode.FALLBACK].layer_name == "SiluAndMul"


@pytest.mark.cuda_only
def test_capability():
    linear = TorchLinearWithCounter(32, 32).to("cuda")
    with use_kernel_mapping(
        {
            "Linear": {
                Device(
                    type="cuda",
                    properties=CUDAProperties(
                        min_capability=75, max_capability=sys.maxsize
                    ),
                ): LayerRepository(
                    repo_id="kernels-test/backward-marker-test",
                    layer_name="LinearBackward",
                )
            }
        }
    ):
        kernelize(linear, mode=Mode.INFERENCE)
        X = torch.randn(10, 32, device="cuda")
        linear(X)

        # Check that we called out to the kernel.
        assert linear.n_calls == 0

    with use_kernel_mapping(
        {
            "Linear": {
                Device(
                    type="cuda",
                    properties=CUDAProperties(
                        min_capability=sys.maxsize, max_capability=sys.maxsize
                    ),
                ): LayerRepository(
                    repo_id="kernels-test/backward-marker-test",
                    layer_name="LinearBackward",
                )
            }
        }
    ):
        kernelize(linear, mode=Mode.INFERENCE)
        X = torch.randn(10, 32, device="cuda")
        linear(X)

        # Check that we didn't call out to the kernel because there is
        # is no kernel with a matching capability..
        assert linear.n_calls == 1


def test_layer_fallback_works():
    @use_kernel_forward_from_hub("SiluAndMulNonExisting")
    class SiluAndMulWithKernelFallback(SiluAndMul):
        pass

    # Check that we don't raise an exception for a non-existing kernel.
    silu_and_mul = SiluAndMulWithKernelFallback()
    kernelize(silu_and_mul, device="cuda", mode=Mode.INFERENCE)


def test_local_layer_repo(device):
    # Fetch a kernel to the local cache.
    package_name, path = install_kernel("kernels-test/backward-marker-test", "main")

    linear = TorchLinearWithCounter(32, 32).to(device)

    with use_kernel_mapping(
        {
            "Linear": {
                device: LocalLayerRepository(
                    # install_kernel will give the fully-resolved path.
                    repo_path=path.parent.parent,
                    package_name=package_name,
                    layer_name="LinearBackward",
                )
            }
        },
        inherit_mapping=False,
    ):
        kernelize(linear, mode=Mode.INFERENCE)

    X = torch.randn(10, 32, device=device)
    linear(X)
    assert linear.n_calls == 0


@pytest.mark.cuda_only
@pytest.mark.parametrize("cls", [SiluAndMulWithKernel, SiluAndMulNoCompileKernel])
@pytest.mark.parametrize("device", ["cuda"])
def test_torch_compile_layer_without_fallback(cls, device):
    silu_and_mul = SiluAndMul()

    X = torch.randn((32, 64), dtype=torch.float32, device=device)
    Y = silu_and_mul(X)

    silu_and_mul_with_kernel = cls()
    silu_and_mul_with_kernel.eval()

    ctx = (
        pytest.raises(ValueError, match="does not support mode")
        if cls is SiluAndMulNoCompileKernel
        else nullcontext()
    )
    with ctx:
        silu_and_mul_with_kernel = kernelize(
            silu_and_mul_with_kernel,
            device=device,
            mode=Mode.INFERENCE | Mode.TORCH_COMPILE,
            use_fallback=False,
        )
    silu_and_mul_compiled = torch.compile(silu_and_mul_with_kernel, fullgraph=True)

    Y_compiled = silu_and_mul_compiled(X)

    torch.testing.assert_close(Y_compiled, Y)


@pytest.mark.cuda_only
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
        mode=Mode.INFERENCE | Mode.TORCH_COMPILE,
    )
    silu_and_mul_compiled = torch.compile(silu_and_mul_with_kernel, fullgraph=True)

    Y_compiled = silu_and_mul_compiled(X)

    torch.testing.assert_close(Y_compiled, Y)


@pytest.mark.cuda_only
def test_mapping_contexts():
    # Make sure we start from scratch.
    register_kernel_mapping(kernel_layer_mapping, inherit_mapping=False)

    assert set(_KERNEL_MAPPING.get().keys()) == {
        "SiluAndMul",
        "SiluAndMulStringDevice",
        "SiluAndMulNoCompile",
        "LigerRMSNorm",
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
            "LigerRMSNorm",
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
                "LigerRMSNorm",
                "TestKernel",
            }
            assert (
                _KERNEL_MAPPING.get()["SiluAndMul"]["cuda"]
                .repos[Mode.FALLBACK]
                ._repo_id
                == "kernels-community/non-existing"
            )

        assert set(_KERNEL_MAPPING.get().keys()) == {
            "SiluAndMul",
            "SiluAndMulStringDevice",
            "SiluAndMulNoCompile",
            "LigerRMSNorm",
            "TestKernel",
        }
        assert (
            _KERNEL_MAPPING.get()["SiluAndMul"]["cuda"].repos[Mode.FALLBACK]._repo_id
            == "kernels-community/activation"
        )

        with use_kernel_mapping(extra_mapping2, inherit_mapping=False):
            assert set(_KERNEL_MAPPING.get().keys()) == {
                "SiluAndMul",
            }
            assert (
                _KERNEL_MAPPING.get()["SiluAndMul"]["cuda"]
                .repos[Mode.FALLBACK]
                ._repo_id
                == "kernels-community/non-existing"
            )

        assert set(_KERNEL_MAPPING.get().keys()) == {
            "SiluAndMul",
            "SiluAndMulStringDevice",
            "SiluAndMulNoCompile",
            "LigerRMSNorm",
            "TestKernel",
        }
        assert (
            _KERNEL_MAPPING.get()["SiluAndMul"]["cuda"].repos[Mode.FALLBACK]._repo_id
            == "kernels-community/activation"
        )

    assert set(_KERNEL_MAPPING.get().keys()) == {
        "SiluAndMul",
        "SiluAndMulStringDevice",
        "SiluAndMulNoCompile",
        "LigerRMSNorm",
    }


def test_validate_kernel_layer():
    class BadLayer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.foo = 42

    def stub_repo(layer):
        return LayerRepository(
            repo_id="kernels-test/nonexisting", layer_name=layer.__name__
        )

    with pytest.raises(
        TypeError,
        match="`kernels-test/nonexisting`.*layer `BadLayer` must not override",
    ):
        _validate_layer(cls=BadLayer, check_cls=SiluAndMul, repo=stub_repo(BadLayer))

    class BadLayer2(nn.Module):
        foo: int = 42

    with pytest.raises(
        TypeError,
        match="`kernels-test/nonexisting`.*layer `BadLayer2` must not contain.*SiluAndMul",
    ):
        _validate_layer(cls=BadLayer2, check_cls=SiluAndMul, repo=stub_repo(BadLayer2))

    class BadLayer3(nn.Module):
        def forward(self, x: torch.Tensor, foo: int) -> torch.Tensor: ...

    with pytest.raises(
        TypeError,
        match="Forward.*`kernels-test/nonexisting`.*layer `BadLayer3` does not match `SiluAndMul`: different number of arguments",
    ):
        _validate_layer(cls=BadLayer3, check_cls=SiluAndMul, repo=stub_repo(BadLayer3))

    class BadLayer4(nn.Module):
        def forward(self, *, x: torch.Tensor) -> torch.Tensor: ...

    with pytest.raises(
        TypeError,
        match="Forward.*`kernels-test/nonexisting`.*layer `BadLayer4` does not match `SiluAndMul`: different kind of arguments",
    ):
        _validate_layer(cls=BadLayer4, check_cls=SiluAndMul, repo=stub_repo(BadLayer4))


@pytest.mark.cuda_only
def test_invalid_mode_for_mapping_rejected():
    linear = TorchLinearWithCounter(32, 32).to("cuda")

    with use_kernel_mapping(
        {
            "Linear": {
                "cuda": {
                    Mode.TRAINING: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearNoBackward",
                    )
                }
            }
        }
    ):
        with pytest.raises(ValueError, match="does not support backward"):
            kernelize(linear, mode=Mode.TRAINING)


@pytest.mark.cuda_only
def test_kernel_modes():
    linear = TorchLinearWithCounter(32, 32).to("cuda")

    # Case 1: layer without further specification, becomes the
    #         base layer.
    with use_kernel_mapping(
        {
            "Linear": {
                "cuda": LayerRepository(
                    repo_id="kernels-test/backward-marker-test",
                    layer_name="LinearBackward",
                )
            }
        }
    ):
        kernelize(linear, mode=Mode.INFERENCE)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        assert linear.n_calls == 0

        kernelize(linear, mode=Mode.TRAINING)
        linear(X)
        assert linear.n_calls == 0

        kernelize(linear, mode=Mode.TRAINING | Mode.TORCH_COMPILE)
        linear(X)
        assert linear.n_calls == 0

    # Case 2: register a kernel just for training. If no base kernel
    #         layer is registered, we fall back to the original layer.
    with use_kernel_mapping(
        {
            "Linear": {
                "cuda": {
                    Mode.TRAINING: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    )
                }
            }
        }
    ):
        kernelize(linear, mode=Mode.INFERENCE)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        assert linear.n_calls == 0

        kernelize(linear, mode=Mode.TRAINING)
        linear(X)
        # Training has a kernel, so fallback.
        assert linear.n_calls == 0

        kernelize(linear, mode=Mode.TRAINING | Mode.TORCH_COMPILE)
        linear(X)
        # TRAINING | TORCH_COMPILE cannot fall back to TRAINING kernel, so uses original.
        assert linear.n_calls == 1

    # Case 3: register a kernel just for training and one for fallback.
    with use_kernel_mapping(
        {
            "Linear": {
                "cuda": {
                    Mode.FALLBACK: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    ),
                    Mode.TRAINING: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    ),
                }
            }
        }
    ):
        kernelize(linear, mode=Mode.INFERENCE)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        # Falls back to TRAINING.
        assert linear.n_calls == 1

        kernelize(linear, mode=Mode.TRAINING)
        linear(X)
        # Falls back to the TRAINING kernel.
        assert linear.n_calls == 1

        kernelize(linear, mode=Mode.TRAINING | Mode.TORCH_COMPILE)
        linear(X)
        # TRAINING | TORCH_COMPILE falls back to FALLBACK kernel.
        assert linear.n_calls == 1

    # Case 4: register a kernel with two preferences.
    with use_kernel_mapping(
        {
            "Linear": {
                "cuda": {
                    Mode.TRAINING
                    | Mode.TORCH_COMPILE: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    )
                }
            }
        }
    ):
        kernelize(linear, mode=Mode.INFERENCE)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        # Falls back to the TRAINING | TORCH_COMPILE kernel.
        assert linear.n_calls == 1

        kernelize(linear, mode=Mode.TRAINING)
        linear(X)
        # TRAINING can fall back to TRAINING | TORCH_COMPILE kernel.
        assert linear.n_calls == 1

        kernelize(linear, mode=Mode.TRAINING | Mode.TORCH_COMPILE)
        linear(X)
        # Uses TRAINING | TORCH_COMPILE kernel.
        assert linear.n_calls == 1


@pytest.mark.cuda_only
def test_fallback_used_when_training():
    linear = TorchLinearWithCounter(32, 32).to("cuda")

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
        kernelize(linear, mode=Mode.INFERENCE)
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
        kernelize(linear, mode=Mode.INFERENCE)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        assert linear.n_calls == 0

        linear.eval()
        linear(X)
        assert linear.n_calls == 0


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _ = Mode.INFERENCE | Mode.TRAINING

    with pytest.raises(ValueError, match="cannot be combined with other modes"):
        _ = Mode.FALLBACK | Mode.TORCH_COMPILE

    with pytest.raises(
        ValueError, match="can only be used to register kernel mappings"
    ):
        kernelize(torch.nn.Linear(32, 32), mode=Mode.FALLBACK)

    with pytest.raises(ValueError, match="mode must contain"):
        kernelize(torch.nn.Linear(32, 32), mode=Mode.TORCH_COMPILE)


@pytest.mark.cuda_only
def test_kernel_modes_inference():
    """Test inference-specific fallback scenarios."""
    linear = TorchLinearWithCounter(32, 32).to("cuda")

    # Case 1: register a kernel just for inference
    with use_kernel_mapping(
        {
            "Linear": {
                "cuda": {
                    Mode.INFERENCE: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    )
                }
            }
        }
    ):
        kernelize(linear, mode=Mode.INFERENCE)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        assert linear.n_calls == 0

        kernelize(linear, mode=Mode.INFERENCE | Mode.TORCH_COMPILE)
        linear(X)
        # INFERENCE | TORCH_COMPILE cannot fall back to INFERENCE kernel, so uses original
        assert linear.n_calls == 1

        kernelize(linear, mode=Mode.TRAINING)
        linear(X)
        # No training kernel, so fallback to original
        assert linear.n_calls == 2

    # Case 2: register a kernel just for inference + torch.compile
    with use_kernel_mapping(
        {
            "Linear": {
                "cuda": {
                    Mode.INFERENCE
                    | Mode.TORCH_COMPILE: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    )
                }
            }
        }
    ):
        kernelize(linear, mode=Mode.INFERENCE | Mode.TORCH_COMPILE)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        assert linear.n_calls == 2

        kernelize(linear, mode=Mode.INFERENCE)
        linear(X)
        # INFERENCE falls back to INFERENCE | TORCH_COMPILE kernel
        assert linear.n_calls == 2

        kernelize(linear, mode=Mode.TRAINING)
        linear(X)
        # No training kernel, so fallback to original
        assert linear.n_calls == 3

    # Case 3: register both inference kernels
    with use_kernel_mapping(
        {
            "Linear": {
                "cuda": {
                    Mode.INFERENCE: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    ),
                    Mode.INFERENCE
                    | Mode.TORCH_COMPILE: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    ),
                }
            }
        }
    ):
        kernelize(linear, mode=Mode.INFERENCE)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        # Uses exact INFERENCE kernel
        assert linear.n_calls == 3

        kernelize(linear, mode=Mode.INFERENCE | Mode.TORCH_COMPILE)
        linear(X)
        # Uses exact INFERENCE | TORCH_COMPILE kernel
        assert linear.n_calls == 3

        kernelize(linear, mode=Mode.TRAINING)
        linear(X)
        # No training kernel, so fallback to original
        assert linear.n_calls == 4


@pytest.mark.cuda_only
def test_kernel_modes_mixed():
    """Test mixed training and inference kernel scenarios."""
    linear = TorchLinearWithCounter(32, 32).to("cuda")

    # Case 1: register both base inference and training kernels
    with use_kernel_mapping(
        {
            "Linear": {
                "cuda": {
                    Mode.INFERENCE: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    ),
                    Mode.TRAINING: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    ),
                }
            }
        }
    ):
        kernelize(linear, mode=Mode.INFERENCE)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        assert linear.n_calls == 0

        kernelize(linear, mode=Mode.TRAINING)
        linear(X)
        assert linear.n_calls == 0

        kernelize(linear, mode=Mode.INFERENCE | Mode.TORCH_COMPILE)
        linear(X)
        # INFERENCE | TORCH_COMPILE cannot fall back to INFERENCE kernel, so uses original
        assert linear.n_calls == 1

        kernelize(linear, mode=Mode.TRAINING | Mode.TORCH_COMPILE)
        linear(X)
        # TRAINING | TORCH_COMPILE cannot fall back to TRAINING kernel, so uses original
        assert linear.n_calls == 2

    # Case 2: register all four kernel modes
    with use_kernel_mapping(
        {
            "Linear": {
                "cuda": {
                    Mode.INFERENCE: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    ),
                    Mode.TRAINING: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    ),
                    Mode.INFERENCE
                    | Mode.TORCH_COMPILE: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    ),
                    Mode.TRAINING
                    | Mode.TORCH_COMPILE: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    ),
                }
            }
        }
    ):
        kernelize(linear, mode=Mode.INFERENCE)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        # Uses exact INFERENCE kernel
        assert linear.n_calls == 2

        kernelize(linear, mode=Mode.TRAINING)
        linear(X)
        # Uses exact TRAINING kernel
        assert linear.n_calls == 2

        kernelize(linear, mode=Mode.INFERENCE | Mode.TORCH_COMPILE)
        linear(X)
        # Uses exact INFERENCE | TORCH_COMPILE kernel
        assert linear.n_calls == 2

        kernelize(linear, mode=Mode.TRAINING | Mode.TORCH_COMPILE)
        linear(X)
        # Uses exact TRAINING | TORCH_COMPILE kernel
        assert linear.n_calls == 2


@pytest.mark.cuda_only
def test_kernel_modes_cross_fallback():
    """Test cross-mode fallback scenarios from inference to training modes."""
    linear = TorchLinearWithCounter(32, 32).to("cuda")

    # Case 1: Only training kernel registered - inference should fall back to training
    with use_kernel_mapping(
        {
            "Linear": {
                "cuda": {
                    Mode.TRAINING: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    )
                }
            }
        }
    ):
        kernelize(linear, mode=Mode.INFERENCE)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        # INFERENCE falls back to TRAINING kernel
        assert linear.n_calls == 0

        kernelize(linear, mode=Mode.TRAINING)
        linear(X)
        # TRAINING uses the kernel directly
        assert linear.n_calls == 0

    # Case 2: Only training + torch.compile kernel registered
    with use_kernel_mapping(
        {
            "Linear": {
                "cuda": {
                    Mode.TRAINING
                    | Mode.TORCH_COMPILE: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    )
                }
            }
        }
    ):
        kernelize(linear, mode=Mode.INFERENCE)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        # INFERENCE falls back to TRAINING | TORCH_COMPILE kernel
        assert linear.n_calls == 0

        kernelize(linear, mode=Mode.INFERENCE | Mode.TORCH_COMPILE)
        linear(X)
        # INFERENCE | TORCH_COMPILE falls back to TRAINING | TORCH_COMPILE kernel
        assert linear.n_calls == 0

        kernelize(linear, mode=Mode.TRAINING)
        linear(X)
        # TRAINING falls back to TRAINING | TORCH_COMPILE kernel
        assert linear.n_calls == 0

        kernelize(linear, mode=Mode.TRAINING | Mode.TORCH_COMPILE)
        linear(X)
        # TRAINING | TORCH_COMPILE uses the kernel directly
        assert linear.n_calls == 0

    # Case 3: Test that training modes don't fall back to inference modes
    with use_kernel_mapping(
        {
            "Linear": {
                "cuda": {
                    Mode.INFERENCE: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    ),
                    Mode.INFERENCE
                    | Mode.TORCH_COMPILE: LayerRepository(
                        repo_id="kernels-test/backward-marker-test",
                        layer_name="LinearBackward",
                    ),
                }
            }
        }
    ):
        kernelize(linear, mode=Mode.TRAINING)
        X = torch.randn(10, 32, device="cuda")
        linear(X)
        # TRAINING should NOT fall back to inference kernels, use original
        assert linear.n_calls == 1

        kernelize(linear, mode=Mode.TRAINING | Mode.TORCH_COMPILE)
        linear(X)
        # TRAINING | TORCH_COMPILE should NOT fall back to inference kernels, use original
        assert linear.n_calls == 2


def test_layer_versions(device):
    @use_kernel_forward_from_hub("Version")
    class Version(nn.Module):
        def forward(self) -> str:
            return "0.0.0"

    version = Version()

    with use_kernel_mapping(
        {
            "Version": {
                Device(type=device): LayerRepository(
                    repo_id="kernels-test/versions",
                    layer_name="Version",
                )
            }
        }
    ):
        version = kernelize(version, device=device, mode=Mode.INFERENCE)
        assert version() == "0.2.0"

    with use_kernel_mapping(
        {
            "Version": {
                Device(type=device): LayerRepository(
                    repo_id="kernels-test/versions",
                    layer_name="Version",
                    version="<1.0.0",
                )
            }
        }
    ):
        version = kernelize(version, device=device, mode=Mode.INFERENCE)
        assert version() == "0.2.0"

    with use_kernel_mapping(
        {
            "Version": {
                Device(type=device): LayerRepository(
                    repo_id="kernels-test/versions",
                    layer_name="Version",
                    version="<0.2.0",
                )
            }
        }
    ):
        version = kernelize(version, device=device, mode=Mode.INFERENCE)
        assert version() == "0.1.1"

    with use_kernel_mapping(
        {
            "Version": {
                Device(type=device): LayerRepository(
                    repo_id="kernels-test/versions",
                    layer_name="Version",
                    version=">0.1.0,<0.2.0",
                )
            }
        }
    ):
        version = kernelize(version, device=device, mode=Mode.INFERENCE)
        assert version() == "0.1.1"

    with use_kernel_mapping(
        {
            "Version": {
                Device(type=device): LayerRepository(
                    repo_id="kernels-test/versions",
                    layer_name="Version",
                    version=">0.2.0",
                )
            }
        }
    ):
        with pytest.raises(ValueError, match=r"No version.*satisfies requirement"):
            kernelize(version, device=device, mode=Mode.INFERENCE)

    with pytest.raises(ValueError, match=r"Either a revision or a version.*not both"):
        use_kernel_mapping(
            {
                "Version": {
                    Device(type=device): LayerRepository(
                        repo_id="kernels-test/versions",
                        layer_name="Version",
                        revision="v0.1.0",
                        version="<1.0.0",
                    )
                }
            }
        )
