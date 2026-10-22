import json
import sys
import types

import pytest

from kernels.importer import _import_from_path, _loaded_kernels

_EXEC_LOG_MODULE = "_kernels_test_exec_log"
_COUNTING_ID = "counting_1_cuda"


def _write_variant(tmp_path):
    variant_dir = tmp_path / "build" / "torch28-cxx11-cu128-x86_64-linux"
    variant_dir.mkdir(parents=True)
    metadata = {
        "id": "broken_1_cuda",
        "name": "broken",
        "version": 1,
        "license": "Apache-2.0",
        "python-depends": ["torch"],
        "backend": {"type": "cuda"},
    }
    (variant_dir / "metadata.json").write_text(json.dumps(metadata))
    (variant_dir / "__init__.py").write_text("raise RuntimeError('kernel is broken')\n")
    return variant_dir


def test_failed_import_cleans_up_sys_modules(tmp_path):
    variant_dir = _write_variant(tmp_path)
    _loaded_kernels.pop(variant_dir, None)
    try:
        with pytest.raises(RuntimeError, match="kernel is broken") as exc_info:
            _import_from_path(variant_dir, deps={})
        assert "broken_1_cuda" not in sys.modules
        assert variant_dir not in _loaded_kernels
        if sys.version_info >= (3, 11):
            assert any("broken" in note for note in exc_info.value.__notes__)
    finally:
        _loaded_kernels.pop(variant_dir, None)
        sys.modules.pop("broken_1_cuda", None)


def _write_counting_variant(base_path, kernel_id):
    """Write a kernel variant that records each execution of its module."""
    variant_dir = base_path / "build" / "torch28-cxx11-cu128-x86_64-linux"
    variant_dir.mkdir(parents=True)
    metadata = {
        "id": kernel_id,
        "name": "counting",
        "version": 1,
        "license": "Apache-2.0",
        "python-depends": ["torch"],
        "backend": {"type": "cuda"},
    }
    (variant_dir / "metadata.json").write_text(json.dumps(metadata))
    (variant_dir / "__init__.py").write_text(f"import {_EXEC_LOG_MODULE}\n{_EXEC_LOG_MODULE}.calls.append(__file__)\n")
    return variant_dir


@pytest.fixture
def exec_log(monkeypatch):
    log = types.ModuleType(_EXEC_LOG_MODULE)
    log.calls = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _EXEC_LOG_MODULE, log)
    return log.calls  # type: ignore[attr-defined]


def test_same_id_different_path_is_not_reloaded(tmp_path, exec_log):
    first_dir = _write_counting_variant(tmp_path / "first", _COUNTING_ID)
    second_dir = _write_counting_variant(tmp_path / "second", _COUNTING_ID)
    try:
        first = _import_from_path(first_dir, deps={})
        second = _import_from_path(second_dir, deps={})

        assert first is second
        assert len(exec_log) == 1
        assert _loaded_kernels[first_dir].module is first
        assert _loaded_kernels[second_dir].module is first
    finally:
        _loaded_kernels.pop(first_dir, None)
        _loaded_kernels.pop(second_dir, None)
        sys.modules.pop(_COUNTING_ID, None)


def test_already_imported_kernel_is_reregistered(tmp_path, exec_log):
    variant_dir = _write_counting_variant(tmp_path, _COUNTING_ID)
    try:
        first = _import_from_path(variant_dir, deps={})
        _loaded_kernels.pop(variant_dir)

        second = _import_from_path(variant_dir, deps={})

        assert first is second
        assert len(exec_log) == 1
        assert _loaded_kernels[variant_dir].module is first
    finally:
        _loaded_kernels.pop(variant_dir, None)
        sys.modules.pop(_COUNTING_ID, None)
