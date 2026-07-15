import json

import pytest
from kernels_data import Metadata

from kernels.utils import _import_from_path, _loaded_kernels, _warn_if_dirty


def _write_variant(tmp_path, provenance):
    variant_dir = tmp_path / "build" / "torch28-cxx11-cu128-x86_64-linux"
    variant_dir.mkdir(parents=True)
    metadata = {
        "id": "activation_1_cuda",
        "name": "activation",
        "version": 1,
        "license": "Apache-2.0",
        "python-depends": ["torch"],
        "backend": {"type": "cuda"},
    }
    if provenance is not None:
        metadata["provenance"] = provenance
    (variant_dir / "metadata.json").write_text(json.dumps(metadata))
    return variant_dir


CLEAN_PROVENANCE = {
    "kernel-builder": {"version": "0.1.0", "sha": "a" * 40, "dirty": False},
    "kernel": {"sha": "b" * 40, "dirty": False},
}
DIRTY_KERNEL = {
    "kernel-builder": {"version": "0.1.0", "sha": "a" * 40, "dirty": False},
    "kernel": {"sha": "b" * 40, "dirty": True},
}
DIRTY_BUILDER = {
    "kernel-builder": {"version": "0.1.0", "sha": "a" * 40, "dirty": True},
    "kernel": {"sha": "b" * 40, "dirty": False},
}


@pytest.mark.parametrize("provenance", [None, CLEAN_PROVENANCE])
def test_no_warning_when_clean(tmp_path, recwarn, provenance):
    variant_dir = _write_variant(tmp_path, provenance)
    metadata = Metadata.read_from_file(variant_dir / "metadata.json")
    _warn_if_dirty(metadata, variant_dir.name)
    assert len(recwarn) == 0


def test_warns_on_dirty_kernel_source(tmp_path):
    variant_dir = _write_variant(tmp_path, DIRTY_KERNEL)
    metadata = Metadata.read_from_file(variant_dir / "metadata.json")
    with pytest.warns(UserWarning, match="dirty git tree"):
        _warn_if_dirty(metadata, variant_dir.name)


def test_warns_names_dirty_sources(tmp_path):
    variant_dir = _write_variant(tmp_path, DIRTY_BUILDER)
    metadata = Metadata.read_from_file(variant_dir / "metadata.json")
    with pytest.warns(UserWarning, match="kernel-builder"):
        _warn_if_dirty(metadata, variant_dir.name)


def test_import_from_path_warns_on_dirty(tmp_path):
    variant_dir = _write_variant(tmp_path, DIRTY_KERNEL)
    (variant_dir / "__init__.py").write_text("value = 42\n")
    _loaded_kernels.pop(variant_dir, None)
    try:
        with pytest.warns(UserWarning, match="dirty git tree"):
            module = _import_from_path(variant_dir)
        assert module.value == 42
    finally:
        _loaded_kernels.pop(variant_dir, None)
