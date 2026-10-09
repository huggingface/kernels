import json
import sys

import pytest

from kernels._rust import Oid
from kernels.hf_hub import RepoInfo
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


@pytest.mark.parametrize("revision", [None, "a" * 40])
def test_failed_import_cleans_up_sys_modules(tmp_path, revision):
    variant_dir = _write_variant(tmp_path)
    repo_info = RepoInfo("test/broken", Oid.from_str(revision)) if revision else None
    import_name = f"broken_1_cuda_{revision}" if revision else "broken_1_cuda"
    _loaded_kernels.pop(variant_dir, None)
    try:
        with pytest.raises(RuntimeError, match="kernel is broken") as exc_info:
            _import_from_path(variant_dir, deps={}, repo_info=repo_info)
        assert import_name not in sys.modules
        assert variant_dir not in _loaded_kernels
        if sys.version_info >= (3, 11):
            assert any("broken" in note for note in exc_info.value.__notes__)
    finally:
        _loaded_kernels.pop(variant_dir, None)
        sys.modules.pop(import_name, None)


def test_revisions_with_same_metadata_id_have_separate_submodules(tmp_path):
    variants = []
    modules = []
    try:
        for revision in ("a" * 40, "b" * 40):
            variant_dir = _write_variant(tmp_path / revision)
            variants.append(variant_dir)
            (variant_dir / "__init__.py").write_text(
                "from . import layers\ndef load_lazy():\n    from . import lazy\n    return lazy\n"
            )
            for filename in ("layers.py", "lazy.py"):
                (variant_dir / filename).write_text(f"revision = {revision!r}\n")
            repo_info = RepoInfo("test/broken", Oid.from_str(revision))
            module = _import_from_path(variant_dir, deps={}, repo_info=repo_info)
            modules.append(module)
            assert _import_from_path(variant_dir, deps={}, repo_info=repo_info) is module

        old, new = modules
        assert old is not new
        assert old.layers is not new.layers
        assert old.layers.revision == "a" * 40
        assert new.layers.revision == "b" * 40
        # Imports made after both revisions are loaded must remain isolated too.
        assert old.load_lazy().revision == "a" * 40
        assert new.load_lazy().revision == "b" * 40
    finally:
        for variant_dir in variants:
            _loaded_kernels.pop(variant_dir, None)
        for module in modules:
            for name in list(sys.modules):
                if name == module.__name__ or name.startswith(f"{module.__name__}."):
                    sys.modules.pop(name, None)
