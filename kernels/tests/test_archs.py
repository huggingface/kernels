import pytest
import torch
from kernels_data import Metadata

from kernels import get_kernel, has_kernel, install_kernel
from kernels.archs import _supports_cuda_capability


@pytest.mark.parametrize(
    "archs,capability,supported",
    [
        # Base archs run on the same generation with the same or a newer
        # minor capability.
        (["8.0"], (8, 0), True),
        (["8.0"], (8, 6), True),
        (["8.6"], (8, 0), False),
        (["8.0"], (7, 5), False),
        (["8.0"], (9, 0), False),
        (["8.0", "9.0"], (9, 0), True),
        # Architecture-specific archs only run on that exact capability.
        (["9.0a"], (9, 0), True),
        (["9.0a"], (9, 1), False),
        (["8.0", "9.0a"], (10, 0), False),
        # Family-specific archs run on the same generation with the same or
        # a newer minor capability.
        (["10.0f"], (10, 0), True),
        (["10.0f"], (10, 3), True),
        (["10.0f"], (12, 0), False),
        # PTX is JIT-compiled for the current device, so builds with `+PTX`
        # also run on any newer capability.
        (["9.0+PTX"], (9, 0), True),
        (["9.0+PTX"], (9, 1), True),
        (["9.0+PTX"], (10, 0), True),
        (["9.0+PTX"], (12, 1), True),
        (["9.0+PTX"], (8, 6), False),
        # ...but the `a` and `f` suffixes keep their own semantics.
        (["9.0a+PTX"], (9, 0), True),
        (["9.0a+PTX"], (10, 0), False),
        (["10.0f+PTX"], (10, 3), True),
        (["10.0f+PTX"], (12, 0), False),
        # Arch strings in an unknown format do not count as a match...
        (["garbage", "8.0"], (9, 0), False),
        # ...but when no arch string can be parsed, compatibility cannot be
        # determined and the build is not rejected.
        (["garbage"], (9, 0), True),
    ],
)
def test_supports_cuda_capability(archs, capability, supported):
    assert _supports_cuda_capability(archs, capability) == supported


@pytest.mark.cuda_only
def test_get_kernel_rejects_unsupported_capability(monkeypatch):
    variant_path = install_kernel("kernels-community/relu", version=1)
    metadata = Metadata.read_from_file(variant_path / "metadata.json")
    if not metadata.backend.archs:
        pytest.skip("kernel build does not declare archs")

    # Fake a capability *below* every declared arch.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (1, 0))

    with pytest.raises(RuntimeError, match="does not support the current device"):
        get_kernel("kernels-community/relu", version=1)
    assert not has_kernel("kernels-community/relu", version=1)

    # The opt-out only checks that a compatible build variant exists.
    assert has_kernel("kernels-community/relu", version=1, check_arch=False)
    kernel = get_kernel("kernels-community/relu", version=1, check_arch=False)
    assert kernel is not None
