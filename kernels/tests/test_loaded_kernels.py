from dataclasses import fields

import pytest

from kernels import get_kernel, get_loaded_kernels, get_local_kernel, install_kernel
from kernels.utils import LoadedKernel, RepoInfos, _loaded_kernels

_REPO_ID = "kernels-test/versions"
_PACKAGE_NAME = "versions"
_VERSION = 2


@pytest.fixture
def fresh_registry():
    """Snapshot the process-wide registry, run the test with a clean one, restore on teardown."""
    saved = _loaded_kernels.copy()
    _loaded_kernels.clear()
    yield
    _loaded_kernels.clear()
    _loaded_kernels.update(saved)


def test_dataclass_shape():
    assert tuple(f.name for f in fields(LoadedKernel)) == (
        "kernel_id",
        "module",
        "module_name",
        "repo_infos",
    )
    assert tuple(f.name for f in fields(RepoInfos)) == ("repo_id", "revision", "backend")


def test_get_loaded_kernels_returns_copy(fresh_registry):
    kernel = get_kernel(_REPO_ID, version=_VERSION, backend="cpu")

    snapshot = get_loaded_kernels()
    assert len(snapshot) == 1

    snapshot.clear()
    snapshot.append("garbage")  # type: ignore[arg-type]

    again = get_loaded_kernels()
    assert len(again) == 1
    assert again[0].module is kernel


def test_get_kernel_registers_loaded_kernel(fresh_registry):
    kernel = get_kernel(_REPO_ID, version=_VERSION, backend="cpu")

    loaded = get_loaded_kernels()
    assert len(loaded) == 1

    entry = loaded[0]
    assert entry.module is kernel
    assert entry.module_name == _PACKAGE_NAME
    assert entry.repo_infos is not None
    assert entry.repo_infos.repo_id == _REPO_ID
    assert isinstance(entry.repo_infos.revision, str) and entry.repo_infos.revision
    assert entry.repo_infos.backend == "cpu"


def test_repeated_get_kernel_is_cached(fresh_registry):
    first = get_kernel(_REPO_ID, version=_VERSION, backend="cpu")
    second = get_kernel(_REPO_ID, version=_VERSION, backend="cpu")

    assert first is second
    assert len(get_loaded_kernels()) == 1


def test_get_local_kernel_registers_with_null_repo_infos(fresh_registry):
    # Populate the HF cache via get_kernel, grab the variant path it registered,
    # then clear the registry and exercise get_local_kernel against that path.
    get_kernel(_REPO_ID, version=_VERSION, backend="cpu")
    (variant_path,) = list(_loaded_kernels.keys())

    _loaded_kernels.clear()

    kernel = get_local_kernel(variant_path, _PACKAGE_NAME, backend="cpu")

    loaded = get_loaded_kernels()
    assert len(loaded) == 1

    entry = loaded[0]
    assert entry.module is kernel
    assert entry.module_name == _PACKAGE_NAME
    assert entry.repo_infos is None


def test_install_kernel_plus_import_does_not_set_repo_infos(fresh_registry):
    # install_kernel alone does not import; it returns a path. Any loader
    # that does not go through get_kernel must leave repo_infos as None.
    package_name, variant_path = install_kernel(_REPO_ID, revision="main", backend="cpu")
    assert package_name == _PACKAGE_NAME
    assert get_loaded_kernels() == []

    get_local_kernel(variant_path, package_name, backend="cpu")
    (entry,) = get_loaded_kernels()
    assert entry.repo_infos is None
