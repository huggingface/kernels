from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import kernels.load as load_module
from kernels import (
    FuncRepository,
    LayerRepository,
    LocalFuncRepository,
    LocalLayerRepository,
    LockedFuncRepository,
    LockedLayerRepository,
    get_kernel,
    get_local_kernel,
    get_locked_kernel,
    has_kernel,
    install_kernel,
    load_kernel,
)
from kernels._rust import Metadata
from kernels.archs import _supports_cuda_capability
from kernels.deps import DepTreeNode
from kernels.resolver import LocalKernel


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


@pytest.fixture(params=[False, True], ids=["online", "offline"])
def kernel_tree(request, monkeypatch, make_metadata, tmp_path):
    monkeypatch.setattr(load_module.constants, "HF_HUB_OFFLINE", request.param)
    for module_name in ("kernels.load", "kernels.layer.layer", "kernels.layer.func"):
        monkeypatch.setattr(f"{module_name}._get_hf_api", lambda **kwargs: None)
    tree = DepTreeNode(
        location=LocalKernel(variant_path=tmp_path, metadata=make_metadata("cuda", ["9.0a"])),
        deps={},
    )
    module = ModuleType("test_kernel")
    module.layers = SimpleNamespace(Version=torch.nn.Identity)
    module.version = lambda: 1
    installed_tree = Mock()
    installed_tree.load.return_value = module
    install = Mock(return_value=installed_tree)
    monkeypatch.setattr(tree, "install", install)
    monkeypatch.setattr(load_module, "resolve_kernel_tree", lambda **kwargs: tree)
    return tree, install


@pytest.fixture(
    params=[
        get_kernel,
        get_local_kernel,
        load_kernel,
        get_locked_kernel,
        LayerRepository,
        FuncRepository,
        LocalLayerRepository,
        LocalFuncRepository,
        LockedLayerRepository,
        LockedFuncRepository,
    ]
)
def kernel_loader(request, tmp_path):
    repo_id = "kernels-test/versions"
    lockfile = Path(__file__).parent / "layer_locking" / "kernels.lock"

    def load(**kwargs):
        if request.param is get_kernel:
            return get_kernel(repo_id, version=2, **kwargs)
        if request.param is get_local_kernel:
            return get_local_kernel(tmp_path, **kwargs)
        if request.param is LayerRepository:
            return LayerRepository(repo_id, version=2, layer_name="Version", **kwargs).load()
        if request.param is FuncRepository:
            return FuncRepository(repo_id, version=2, func_name="version", **kwargs).load()
        if request.param is LocalLayerRepository:
            return LocalLayerRepository(tmp_path, layer_name="Version", **kwargs).load()
        if request.param is LocalFuncRepository:
            return LocalFuncRepository(tmp_path, func_name="version", **kwargs).load()
        if request.param is LockedLayerRepository:
            return LockedLayerRepository(repo_id, lockfile=lockfile, layer_name="Version", **kwargs).load()
        if request.param is LockedFuncRepository:
            return LockedFuncRepository(repo_id, lockfile=lockfile, func_name="version", **kwargs).load()
        return request.param(repo_id, lockfile=lockfile, **kwargs)

    return load


@pytest.mark.parametrize("in_dependency", [False, True], ids=["root", "dependency"])
def test_loaders_reject_unsupported_capability(
    kernel_loader, kernel_tree, fake_cuda_device, make_metadata, in_dependency
):
    tree, install = kernel_tree
    if in_dependency:
        tree.deps["kernels-test/dependency"] = DepTreeNode(location=tree.location, deps={})
        tree.location = LocalKernel(
            variant_path=tree.location.variant_path,
            metadata=make_metadata("cuda", ["10.0a"]),
        )

    with pytest.raises(RuntimeError, match="CUDA capability 10.0.*not supported"):
        kernel_loader()
    install.assert_not_called()


@pytest.mark.parametrize("archs", [["10.0a"], None], ids=["supported", "undeclared"])
def test_loaders_accept_compatible_archs(kernel_loader, kernel_tree, fake_cuda_device, make_metadata, archs):
    tree, install = kernel_tree
    tree.location = LocalKernel(variant_path=tree.location.variant_path, metadata=make_metadata("cuda", archs))

    assert kernel_loader() is not None
    install.assert_called_once()
    install.return_value.load.assert_called_once()


def test_loaders_can_skip_arch_check(kernel_loader, kernel_tree, fake_cuda_device):
    _, install = kernel_tree

    assert kernel_loader(check_arch=False) is not None
    install.assert_called_once()
    install.return_value.load.assert_called_once()


def test_loaders_skip_arch_check_preserves_other_validation(
    kernel_loader, kernel_tree, fake_cuda_device, make_metadata
):
    tree, install = kernel_tree
    tree.location = LocalKernel(
        variant_path=tree.location.variant_path,
        metadata=make_metadata("cuda", ["9.0a"], kernels_minver="999.1.0"),
    )
    with pytest.raises(RuntimeError, match="requires kernels>=999.1.0"):
        kernel_loader(check_arch=False)
    install.assert_not_called()
