import logging

import pytest
import torch
import torch.nn.functional as F
from huggingface_hub.errors import HfHubHTTPError

from kernels import get_kernel, get_local_kernel, has_kernel, install_kernel


@pytest.fixture
def kernel():
    return get_kernel("kernels-community/relu", version=1)


@pytest.fixture
def local_kernel_path():
    # install_kernel works with resolved revisions, so explicitly use v1 here.
    return install_kernel("kernels-community/relu", revision="v1")


@pytest.fixture
def local_kernel(local_kernel_path):
    path = local_kernel_path
    return get_local_kernel(path.parent.parent)


@pytest.fixture
def metal_kernel():
    return get_kernel("kernels-test/relu-metal")


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    return "cuda"


@pytest.mark.cuda_only
def test_relu(kernel, device):
    x = torch.arange(-4, 5, dtype=torch.float32, device=device).view(3, 3)
    y = kernel.relu(x)
    torch.testing.assert_close(y, F.relu(x))


@pytest.mark.cuda_only
def test_local_kernel(local_kernel, device):
    x = torch.arange(-4, 5, dtype=torch.float32, device=device).view(3, 3)
    y = local_kernel.relu(x)
    torch.testing.assert_close(y, F.relu(x))


@pytest.mark.parametrize(
    "repo_revision",
    [
        ("kernels-test/flattened-build", "pre-flattening"),
        ("kernels-test/flattened-build", "main"),
        ("kernels-test/flattened-build", "without-compat-module"),
    ],
)
def test_local_kernel_path_types(repo_revision, device):
    repo_id, revision = repo_revision
    path = install_kernel(repo_id, revision=revision)

    # Top-level repo path
    # ie: /home/ubuntu/.cache/huggingface/hub/models--kernels-community--activation/snapshots/2fafa6a3a38ccb57a1a98419047cf7816ecbc071
    kernel = get_local_kernel(path.parent.parent)
    x = torch.arange(0, 32, dtype=torch.float16, device=device).view(2, 16)
    torch.testing.assert_close(kernel.silu_and_mul(x), silu_and_mul_torch(x))

    # Build directory path
    # ie: /home/ubuntu/.cache/huggingface/hub/models--kernels-community--activation/snapshots/2fafa6a3a38ccb57a1a98419047cf7816ecbc071/build
    kernel = get_local_kernel(path.parent.parent / "build")
    torch.testing.assert_close(kernel.silu_and_mul(x), silu_and_mul_torch(x))

    # Explicit package path
    # ie: /home/ubuntu/.cache/huggingface/hub/models--kernels-community--activation/snapshots/2fafa6a3a38ccb57a1a98419047cf7816ecbc071/build/torch28-cxx11-cu128-x86_64-linux
    kernel = get_local_kernel(path)
    torch.testing.assert_close(kernel.silu_and_mul(x), silu_and_mul_torch(x))


@pytest.mark.darwin_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_relu_metal(metal_kernel, dtype):
    x = torch.arange(-10, 10, dtype=dtype, device="mps")
    y = metal_kernel.relu(x)
    assert torch.allclose(y, torch.relu(x))


@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "kernel_exists",
    [
        ("kernels-community/relu", "main", True),
        ("kernels-test/silu-and-mul", "v1", True),
        # Repo only contains Torch 2.4 kernels (and we don't
        # support/test against this version).
        ("kernels-test/only-torch-2.4", "main", False),
        ("kernels-test/flattened-build", "main", True),
        ("kernels-test/flattened-build", "without-compat-module", True),
    ],
)
def test_has_kernel(kernel_exists):
    repo_id, revision, kernel = kernel_exists
    assert has_kernel(repo_id, revision=revision) == kernel


def test_version():
    kernel = get_kernel("kernels-test/versions", version=1)
    assert kernel.version() == 1
    kernel = get_kernel("kernels-test/versions", version=2)
    assert kernel.version() == 2

    with pytest.raises(ValueError, match="Version 0 not found, available versions: 1, 2.*"):
        kernel = get_kernel("kernels-test/versions", version=0)


def test_version_outdated_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="kernels._versions"):
        kernel = get_kernel("kernels-test/versions", version=1)
    assert kernel.version() == 1
    assert "You are using version 1 of 'kernels-test/versions', but version 2 is available." in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="kernels._versions"):
        kernel = get_kernel("kernels-test/versions", version=2)
    assert kernel.version() == 2
    assert "but version" not in caplog.text


def test_no_version_or_revision_warning():
    from packaging.version import Version

    from kernels import __version__

    assert Version(__version__) < Version("0.15"), (
        "The deprecation cycle for requiring `version` or `revision` is complete. "
        "Remove the fallback to 'main' in `select_revision_or_version` and make "
        "`version` or `revision` a required argument."
    )
    with pytest.warns(FutureWarning, match="will require specifying a kernel version or revision"):
        get_kernel("kernels-test/versions")


def test_noarch_kernel(device):
    supported_devices = ["cpu", "cuda", "xpu"]
    if device not in supported_devices:
        pytest.skip(f"Device is not one of: {','.join(supported_devices)}")
    get_kernel("kernels-test/silu-and-mul", version=1)


def test_get_kernel_with_backend(device):
    x = torch.randn((16, 16), device=device)
    assert has_kernel("kernels-community/relu", version=1)
    relu = get_kernel("kernels-community/relu", version=1)
    torch.testing.assert_close(relu.relu(x), F.relu(x))

    assert has_kernel("kernels-community/relu", version=1, backend=device)
    relu = get_kernel("kernels-community/relu", version=1, backend=device)
    torch.testing.assert_close(relu.relu(x), F.relu(x))

    with pytest.raises(ValueError, match="Invalid backend 'xpu'"):
        get_kernel("kernels-community/relu", version=1, backend="xpu")

    assert has_kernel("kernels-community/relu", version=1, backend="cpu")
    relu = get_kernel("kernels-community/relu", version=1, backend="cpu")
    x = x.cpu()
    torch.testing.assert_close(relu.relu(x), F.relu(x))


@pytest.mark.parametrize(
    "repo_revision",
    [
        ("kernels-test/flattened-build", "pre-flattening"),
        ("kernels-test/flattened-build", "main"),
        ("kernels-test/flattened-build", "without-compat-module"),
    ],
)
def test_flattened_build(repo_revision, device):
    repo_id, revision = repo_revision
    kernel = get_kernel(repo_id, revision=revision)

    x = torch.arange(0, 32, dtype=torch.float16, device=device).view(2, 16)
    torch.testing.assert_close(kernel.silu_and_mul(x), silu_and_mul_torch(x))


def test_local_overrides(monkeypatch, local_kernel_path):
    kernel_path = local_kernel_path

    # Ensure that we are testing with a non-existing kernel, so that we know
    # that the kernel must be local.
    with pytest.raises(HfHubHTTPError):
        get_kernel("kernels-test/activation", revision="main")

    with monkeypatch.context() as m:
        m.setenv(
            "LOCAL_KERNELS",
            f"kernels-test/activation={str(kernel_path)}:kernels-test/non-existing2=/non/existing",
        )
        get_kernel("kernels-test/activation")

    with monkeypatch.context() as m:
        m.setenv(
            "LOCAL_KERNELS",
            f"kernels-test/non-existing2=/non/existing:kernels-test/activation={str(kernel_path)}",
        )
        get_kernel("kernels-test/activation")

    with monkeypatch.context() as m:
        # Using a non-existing path should error.
        m.setenv(
            "LOCAL_KERNELS",
            "kernels-test/non-existing2=/non/existing:kernels-test/activation=/non/existing",
        )
        with pytest.raises(FileNotFoundError, match=r"Could not find kernel in /non/existing"):
            get_kernel("kernels-test/activation")

    with monkeypatch.context() as m:
        # Malformed entries must be rejected.
        m.setenv(
            "LOCAL_KERNELS",
            "kernels-test/non-existing2=/non/existing:kernels-test/activation",
        )
        with pytest.raises(ValueError, match=r"Invalid LOCAL_KERNELS entry"):
            get_kernel("kernels-test/activation")


@pytest.mark.neuron_only
def test_neuron():
    relu = get_kernel("kernels-test/relu-nki", version=1)
    x = torch.randn((16, 16), dtype=torch.float16).to(device="neuron")
    torch.testing.assert_close(relu.relu(x), x.relu())


def test_trust_remote_code_blocks_untrusted_org():
    """Kernels from untrusted orgs should be rejected by default."""
    with pytest.raises(ValueError, match=r"not from a trusted org"):
        get_kernel("drbh/not-a-trused-org-kernel", version=1)


def test_trust_remote_code_allows_trusted_org():
    """Kernels from trusted orgs should not raise a trust error.

    This only checks that the trust check passes — it may still fail
    for other reasons (e.g. network, missing variant).
    """
    # kernels-community is in TRUSTED_KERNEL_ORGS, so trust check passes.
    # The call may fail later (no CUDA, etc.), but not with a trust error.
    get_kernel("kernels-community/relu", version=1)


def test_trust_remote_code_flag_allows_untrusted():
    """trust_remote_code=True should bypass the org check."""
    # This will likely fail for network/variant reasons, but NOT with trust error.
    get_kernel("drbh/ci-test-kernel", version=1, trust_remote_code=True)


def test_trust_remote_code_list_warns_and_falls_back():
    """trust_remote_code as a list should warn and fall back to the default check."""
    with pytest.warns(match=r"Signing identity verification is not yet implemented"):
        with pytest.raises(ValueError, match=r"not from a trusted organisation"):
            get_kernel("some-random-org/some-kernel", version=1, trust_remote_code=["identity-1"])


def silu_and_mul_torch(x: torch.Tensor):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]
