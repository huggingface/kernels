import json

import pytest
from kernels_data import Metadata

from kernels import __version__
from kernels.importer import (
    _import_from_path,
    _loaded_kernels,
    _warn_if_below_minver,
)


def _write_variant(tmp_path, minver):
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
    if minver is not None:
        metadata["minver"] = minver
    (variant_dir / "metadata.json").write_text(json.dumps(metadata))
    return variant_dir


@pytest.mark.parametrize("minver", [None, "0.0.1"])
def test_no_warning_when_minver_met(tmp_path, recwarn, minver):
    variant_dir = _write_variant(tmp_path, minver)
    metadata = Metadata.read_from_file(variant_dir / "metadata.json")
    _warn_if_below_minver(metadata, variant_dir.name)
    assert len(recwarn) == 0


def test_warns_when_minver_not_met(tmp_path):
    variant_dir = _write_variant(tmp_path, "999.0.0")
    metadata = Metadata.read_from_file(variant_dir / "metadata.json")
    with pytest.warns(UserWarning, match="requires kernels>=999.0.0"):
        _warn_if_below_minver(metadata, variant_dir.name)


def test_warning_mentions_installed_version(tmp_path):
    variant_dir = _write_variant(tmp_path, "999.0.0")
    metadata = Metadata.read_from_file(variant_dir / "metadata.json")
    with pytest.warns(UserWarning, match=f"version {__version__} is installed"):
        _warn_if_below_minver(metadata, variant_dir.name)


def test_import_from_path_warns_on_unmet_minver(tmp_path):
    variant_dir = _write_variant(tmp_path, "999.0.0")
    (variant_dir / "__init__.py").write_text("value = 42\n")
    _loaded_kernels.pop(variant_dir, None)
    try:
        with pytest.warns(UserWarning, match="requires kernels>=999.0.0"):
            module = _import_from_path(variant_dir, deps={})
        assert module.value == 42
    finally:
        _loaded_kernels.pop(variant_dir, None)
