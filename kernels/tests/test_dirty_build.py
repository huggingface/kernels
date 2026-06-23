import json
import warnings

import pytest
from kernels_data import Metadata

from kernels.utils import RepoInfo, _import_from_path, _loaded_kernels, _warn_if_dirty_build


def _metadata(build_info: dict | None) -> Metadata:
    data: dict = {
        "name": "relu",
        "id": "_relu_cuda_abc1234",
        "version": 1,
        "license": "Apache-2.0",
        "python-depends": [],
        "backend": {"type": "cuda"},
    }
    if build_info is not None:
        data["build-info"] = build_info
    return Metadata.from_bytes(json.dumps(data).encode("utf-8"))


def _clean_builder() -> dict:
    return {"version": "0.16.0-dev0", "sha": "a" * 40, "dirty": False}


def _dirty_builder() -> dict:
    return {"version": "0.16.0-dev0", "sha": "a" * 40, "dirty": True}


def _clean_kernel() -> dict:
    return {"sha": "b" * 40, "dirty": False}


def _dirty_kernel() -> dict:
    return {"sha": "b" * 40, "dirty": True}


@pytest.fixture
def fresh_registry():
    """Run the test against a clean loaded-kernel registry, restore on teardown."""
    saved = _loaded_kernels.copy()
    _loaded_kernels.clear()
    yield
    _loaded_kernels.clear()
    _loaded_kernels.update(saved)


def _warnings_for(metadata: Metadata, repo_info: RepoInfo | None = None) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_dirty_build(metadata, repo_info)
    return [str(w.message) for w in caught]


def test_no_build_info_does_not_warn():
    assert _warnings_for(_metadata(None)) == []


def test_clean_build_does_not_warn():
    build_info = {"kernel-builder": _clean_builder(), "kernel": _clean_kernel()}
    assert _warnings_for(_metadata(build_info)) == []


def test_dirty_kernel_builder_warns():
    build_info = {"kernel-builder": _dirty_builder(), "kernel": _clean_kernel()}
    messages = _warnings_for(_metadata(build_info))
    assert len(messages) == 1
    assert "`kernel-builder` had uncommitted changes" in messages[0]
    assert "kernel source had uncommitted changes" not in messages[0]


def test_dirty_kernel_source_warns():
    build_info = {"kernel-builder": _clean_builder(), "kernel": _dirty_kernel()}
    messages = _warnings_for(_metadata(build_info))
    assert len(messages) == 1
    assert "kernel source had uncommitted changes" in messages[0]
    assert "`kernel-builder` had uncommitted changes" not in messages[0]


def test_both_dirty_warns_with_both_reasons():
    build_info = {"kernel-builder": _dirty_builder(), "kernel": _dirty_kernel()}
    messages = _warnings_for(_metadata(build_info))
    assert len(messages) == 1
    assert "`kernel-builder` had uncommitted changes" in messages[0]
    assert "kernel source had uncommitted changes" in messages[0]


def test_warning_includes_repo_id_when_available():
    build_info = {"kernel-builder": _dirty_builder(), "kernel": _clean_kernel()}
    messages = _warnings_for(_metadata(build_info), RepoInfo(repo_id="acme/relu", revision="main"))
    assert "acme/relu" in messages[0]


def test_import_from_path_warns_on_dirty_build(tmp_path, fresh_registry):
    variant_path = tmp_path / "variant"
    (variant_path / "relu").mkdir(parents=True)
    (variant_path / "relu" / "__init__.py").write_text("VALUE = 42\n")

    metadata = {
        "name": "relu",
        "id": "_relu_cuda_abc1234",
        "version": 1,
        "license": "Apache-2.0",
        "python-depends": [],
        "backend": {"type": "cuda"},
        "build-info": {"kernel-builder": _dirty_builder(), "kernel": _clean_kernel()},
    }
    (variant_path / "metadata.json").write_text(json.dumps(metadata))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = _import_from_path(variant_path)

    assert module.VALUE == 42
    dirty_warnings = [w for w in caught if "dirty source tree" in str(w.message)]
    assert len(dirty_warnings) == 1
