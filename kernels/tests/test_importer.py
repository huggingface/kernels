import importlib
import json
import py_compile
import sys

import pytest

from kernels.importer import _import_from_path, _loaded_kernels


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


def _plant_pyc(source_path, payload, tmp_path):
    # An unchecked-hash .pyc (PEP 552) is used by the default loader without
    # comparing it against the source.
    evil_src = tmp_path / f"evil_{source_path.stem}.py"
    evil_src.write_text(payload)
    py_compile.compile(
        str(evil_src),
        cfile=importlib.util.cache_from_source(str(source_path)),
        dfile=str(source_path),
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )


def test_import_ignores_bytecode(tmp_path):
    variant_dir = _write_variant(tmp_path)
    (variant_dir / "__init__.py").write_text("from ._sub import WHO as SUB_WHO\nWHO = 'source'\n")
    (variant_dir / "_sub.py").write_text("WHO = 'source'\n")
    _plant_pyc(variant_dir / "__init__.py", "from ._sub import WHO as SUB_WHO\nWHO = 'pyc'\n", tmp_path)
    _plant_pyc(variant_dir / "_sub.py", "WHO = 'pyc'\n", tmp_path)
    # Sourceless bytecode must not be importable either.
    py_compile.compile(str(variant_dir / "_sub.py"), cfile=str(variant_dir / "_payload.pyc"))

    _loaded_kernels.pop(variant_dir, None)
    try:
        module = _import_from_path(variant_dir, deps={})
        assert module.WHO == "source"
        assert module.SUB_WHO == "source"
        with pytest.raises(ImportError):
            importlib.import_module("broken_1_cuda._payload")
    finally:
        _loaded_kernels.pop(variant_dir, None)
        for name in [m for m in sys.modules if m.startswith("broken_1_cuda")]:
            sys.modules.pop(name)
