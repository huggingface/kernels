import json
import re

import pytest
from kernels_data import Metadata, Version

import kernels
from kernels.importer import (
    _import_from_path,
    _installed_version,
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
        metadata["kernels-minver"] = minver
    (variant_dir / "metadata.json").write_text(json.dumps(metadata))
    return variant_dir


@pytest.mark.parametrize("minver", [None, "0.0.1"])
def test_no_warning_when_minver_met(tmp_path, recwarn, minver):
    variant_dir = _write_variant(tmp_path, minver)
    metadata = Metadata.read_from_file(variant_dir / "metadata.json")
    _warn_if_below_minver(metadata, variant_dir.name)
    assert len(recwarn) == 0


def test_no_warning_for_dev_version_of_required_release(tmp_path, recwarn, monkeypatch):
    # A development version implements the release it leads up to, so
    # `0.17.0.dev0` must satisfy a `0.17.0` requirement.
    monkeypatch.setattr(kernels, "__version__", "0.17.0.dev0")
    variant_dir = _write_variant(tmp_path, "0.17.0")
    metadata = Metadata.read_from_file(variant_dir / "metadata.json")
    _warn_if_below_minver(metadata, variant_dir.name)
    assert len(recwarn) == 0


@pytest.mark.parametrize(
    "installed",
    ["0.17.0.dev0", "0.17.0rc1", "0.17.0.post1", "0.17.0+cu121", "0.17"],
)
def test_installed_version_uses_release_segment(monkeypatch, installed):
    monkeypatch.setattr(kernels, "__version__", installed)
    assert _installed_version() == Version.from_str("0.17.0")


def test_installed_version_is_none_for_non_pep440_version(monkeypatch):
    monkeypatch.setattr(kernels, "__version__", "0.17.0-dirty")
    assert _installed_version() is None


def test_unparseable_installed_version_does_not_fail_load(tmp_path, recwarn, monkeypatch):
    # A version that cannot be compared must not turn this advisory check into
    # a hard failure.
    monkeypatch.setattr(kernels, "__version__", "0.17.0-dirty")
    variant_dir = _write_variant(tmp_path, "999.1.0")
    metadata = Metadata.read_from_file(variant_dir / "metadata.json")
    _warn_if_below_minver(metadata, variant_dir.name)
    assert len(recwarn) == 0


def test_version_ordering_is_numeric_not_lexicographic():
    # `0.9 < 0.10` only holds for numeric comparison; string comparison would
    # get this backwards.
    assert Version.from_str("0.9") < Version.from_str("0.10")
    assert Version.from_str("0.14") == Version.from_str("0.14.0")
    assert Version.from_str("0.14.0") < Version.from_str("0.14.1")


def test_warns_when_minver_not_met(tmp_path):
    variant_dir = _write_variant(tmp_path, "999.1.0")
    metadata = Metadata.read_from_file(variant_dir / "metadata.json")
    with pytest.warns(UserWarning, match="requires kernels>=999.1"):
        _warn_if_below_minver(metadata, variant_dir.name)


def test_warning_mentions_installed_version(tmp_path):
    variant_dir = _write_variant(tmp_path, "999.1.0")
    metadata = Metadata.read_from_file(variant_dir / "metadata.json")
    with pytest.warns(UserWarning, match=f"version {re.escape(kernels.__version__)} is installed"):
        _warn_if_below_minver(metadata, variant_dir.name)


def test_import_from_path_warns_on_unmet_minver(tmp_path):
    variant_dir = _write_variant(tmp_path, "999.1.0")
    (variant_dir / "__init__.py").write_text("value = 42\n")
    _loaded_kernels.pop(variant_dir, None)
    try:
        with pytest.warns(UserWarning, match="requires kernels>=999.1"):
            module = _import_from_path(variant_dir, deps={})
        assert module.value == 42
    finally:
        _loaded_kernels.pop(variant_dir, None)
