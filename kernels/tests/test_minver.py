import json
import re
from pathlib import Path

import pytest
from kernels_data import Metadata, Version

import kernels
from kernels.deps import DepTreeNode, MinverValidator, _installed_version
from kernels.resolver import LocalKernel


def _make_tree(minver):
    metadata_dict = {
        "id": "activation_1_cuda",
        "name": "activation",
        "version": 1,
        "license": "Apache-2.0",
        "python-depends": ["torch"],
        "backend": {"type": "cuda"},
    }
    if minver is not None:
        metadata_dict["kernels-minver"] = minver
    metadata = Metadata.from_bytes(json.dumps(metadata_dict).encode("utf-8"))
    variant_path = Path("build") / "torch28-cxx11-cu128-x86_64-linux"
    return DepTreeNode(location=LocalKernel(variant_path, metadata), deps={})


@pytest.mark.parametrize("minver", [None, "0.0.1"])
def test_no_error_when_minver_met(minver):
    MinverValidator().validate(tree=_make_tree(minver))


def test_no_error_for_dev_version_of_required_release(monkeypatch):
    # A development version implements the release it leads up to, so
    # `0.17.0.dev0` must satisfy a `0.17.0` requirement.
    monkeypatch.setattr(kernels, "__version__", "0.17.0.dev0")
    MinverValidator().validate(tree=_make_tree("0.17.0"))


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


def test_unparseable_installed_version_does_not_fail_validation(monkeypatch):
    # A version that cannot be compared must not turn this check into a
    # failure.
    monkeypatch.setattr(kernels, "__version__", "0.17.0-dirty")
    MinverValidator().validate(tree=_make_tree("999.1.0"))


def test_version_ordering_is_numeric_not_lexicographic():
    # `0.9 < 0.10` only holds for numeric comparison; string comparison would
    # get this backwards.
    assert Version.from_str("0.9") < Version.from_str("0.10")
    assert Version.from_str("0.14") == Version.from_str("0.14.0")
    assert Version.from_str("0.14.0") < Version.from_str("0.14.1")


def test_raises_when_minver_not_met():
    with pytest.raises(RuntimeError, match="requires kernels>=999.1"):
        MinverValidator().validate(tree=_make_tree("999.1.0"))


def test_error_mentions_installed_version():
    with pytest.raises(RuntimeError, match=f"version {re.escape(kernels.__version__)} is installed"):
        MinverValidator().validate(tree=_make_tree("999.1.0"))
