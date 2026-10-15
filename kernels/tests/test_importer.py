import json
import sys
import threading
import types

import pytest

from kernels._rust import Digest, DigestAlgorithm
from kernels.importer import _import_from_path, _loaded_kernel_metadata, _loaded_kernels

_EXEC_LOG_MODULE = "_kernels_test_exec_log"
_COUNTING_ID = "counting_1_cuda"
_REBUILT_ID = "counting_2_cuda"


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
        _loaded_kernel_metadata.pop("broken_1_cuda", None)
        sys.modules.pop("broken_1_cuda", None)


def _write_counting_variant(
    base_path,
    kernel_id,
    *,
    variant_tag: str = "",
    with_digest: bool = False,
    init_delay: float = 0.0,
):
    """Write a kernel variant that records each execution of its module.

    Variants with a different `variant_tag` have different files. The module
    sleeps for `init_delay` seconds before it is fully initialized, which is
    the case once `TAG` is set.
    """
    variant_dir = base_path / "build" / "torch28-cxx11-cu128-x86_64-linux"
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "__init__.py").write_text(
        "import time\n"
        f"import {_EXEC_LOG_MODULE}\n"
        f"{_EXEC_LOG_MODULE}.calls.append(__file__)\n"
        f"time.sleep({init_delay!r})\n"
        f"TAG = {variant_tag!r}\n"
    )
    metadata = {
        "id": kernel_id,
        "name": "counting",
        "version": 1,
        "license": "Apache-2.0",
        "python-depends": ["torch"],
        "backend": {"type": "cuda"},
    }
    if with_digest:
        metadata["digest"] = {
            "algorithm": "sha256",
            "files": Digest.hash_variant(DigestAlgorithm.SHA256, variant_dir).files,
        }
    (variant_dir / "metadata.json").write_text(json.dumps(metadata))
    return variant_dir


@pytest.fixture
def exec_log(monkeypatch):
    log = types.ModuleType(_EXEC_LOG_MODULE)
    log.calls = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _EXEC_LOG_MODULE, log)
    return log.calls  # type: ignore[attr-defined]


@pytest.fixture
def counting_cleanup():
    """Remove every registration of the counting kernels after the test."""
    yield
    kernel_ids = {_COUNTING_ID, _REBUILT_ID}
    for path in [path for path, loaded in _loaded_kernels.items() if loaded.metadata.id in kernel_ids]:
        _loaded_kernels.pop(path)
    for kernel_id in kernel_ids:
        _loaded_kernel_metadata.pop(kernel_id, None)
        sys.modules.pop(kernel_id, None)


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
        _loaded_kernel_metadata.pop(_COUNTING_ID, None)
        sys.modules.pop(_COUNTING_ID, None)


def test_same_id_same_digest_is_reused(tmp_path, exec_log, counting_cleanup):
    first_dir = _write_counting_variant(tmp_path / "first", _COUNTING_ID, with_digest=True)
    second_dir = _write_counting_variant(tmp_path / "second", _COUNTING_ID, with_digest=True)

    first = _import_from_path(first_dir, deps={})
    second = _import_from_path(second_dir, deps={})

    assert first is second
    assert len(exec_log) == 1
    assert _loaded_kernels[first_dir].module is first
    assert _loaded_kernels[second_dir].module is first


def test_same_id_different_digest_raises(tmp_path, exec_log, counting_cleanup):
    first_dir = _write_counting_variant(tmp_path / "first", _COUNTING_ID, with_digest=True)
    second_dir = _write_counting_variant(tmp_path / "second", _COUNTING_ID, variant_tag="hacked", with_digest=True)

    _import_from_path(first_dir, deps={})
    with pytest.raises(RuntimeError, match="different build") as exc_info:
        _import_from_path(second_dir, deps={})

    message = str(exc_info.value)
    assert str(second_dir) in message
    assert _COUNTING_ID in message
    assert "__init__.py" in message
    assert len(exec_log) == 1
    assert second_dir not in _loaded_kernels


def test_same_id_check_does_not_depend_on_loaded_kernels(tmp_path, exec_log, counting_cleanup):
    first_dir = _write_counting_variant(tmp_path / "first", _COUNTING_ID, with_digest=True)
    second_dir = _write_counting_variant(tmp_path / "second", _COUNTING_ID, variant_tag="hacked", with_digest=True)

    _import_from_path(first_dir, deps={})
    # The registry of loaded kernel paths does not provide the reference build.
    _loaded_kernels.pop(first_dir)

    with pytest.raises(RuntimeError, match="different build"):
        _import_from_path(second_dir, deps={})


@pytest.mark.parametrize(
    ("first_has_digest", "second_has_digest"),
    [(True, False), (False, True)],
    ids=["only-first-has-digest", "only-second-has-digest"],
)
def test_same_id_one_sided_digest_raises(tmp_path, exec_log, counting_cleanup, first_has_digest, second_has_digest):
    # The files are identical, only the presence of a digest differs.
    first_dir = _write_counting_variant(tmp_path / "first", _COUNTING_ID, with_digest=first_has_digest)
    second_dir = _write_counting_variant(tmp_path / "second", _COUNTING_ID, with_digest=second_has_digest)

    _import_from_path(first_dir, deps={})
    with pytest.raises(RuntimeError, match="has a digest, but"):
        _import_from_path(second_dir, deps={})

    assert len(exec_log) == 1
    assert second_dir not in _loaded_kernels


def test_in_place_rebuild_with_same_id_raises(tmp_path, exec_log, counting_cleanup):
    variant_dir = _write_counting_variant(tmp_path, _COUNTING_ID, with_digest=True)
    _import_from_path(variant_dir, deps={})

    # Modify the kernel in place, including its digest, but keep its id.
    _write_counting_variant(tmp_path, _COUNTING_ID, variant_tag="hacked", with_digest=True)

    with pytest.raises(RuntimeError, match="different build"):
        _import_from_path(variant_dir, deps={})
    assert len(exec_log) == 1


def test_in_place_rebuild_with_new_id_is_loaded(tmp_path, exec_log, counting_cleanup):
    variant_dir = _write_counting_variant(tmp_path, _COUNTING_ID, with_digest=True)
    first = _import_from_path(variant_dir, deps={})

    # A proper rebuild gets a new kernel id.
    _write_counting_variant(tmp_path, _REBUILT_ID, variant_tag="rebuilt", with_digest=True)
    second = _import_from_path(variant_dir, deps={})

    assert second is not first
    assert second.TAG == "rebuilt"
    assert len(exec_log) == 2
    assert _loaded_kernels[variant_dir].module is second


def _load_concurrently(paths):
    """Import the given kernel paths from one thread each, started at the same time."""
    barrier = threading.Barrier(len(paths))
    modules = [None] * len(paths)
    errors = []

    def load(index, path):
        barrier.wait()
        try:
            modules[index] = _import_from_path(path, deps={})
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=load, args=(index, path)) for index, path in enumerate(paths)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads), "kernel import did not finish"
    assert errors == []
    return modules


@pytest.mark.parametrize("same_path", [True, False], ids=["same-path", "same-id-other-path"])
def test_concurrent_imports_execute_kernel_once(tmp_path, exec_log, counting_cleanup, same_path):
    first_dir = _write_counting_variant(tmp_path / "first", _COUNTING_ID, with_digest=True, init_delay=0.2)
    second_dir = (
        first_dir
        if same_path
        else _write_counting_variant(tmp_path / "second", _COUNTING_ID, with_digest=True, init_delay=0.2)
    )

    first, second = _load_concurrently([first_dir, second_dir])

    assert len(exec_log) == 1
    assert first is second
    # Neither thread got a partially initialized module.
    assert first.TAG == ""


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
        _loaded_kernel_metadata.pop(_COUNTING_ID, None)
        sys.modules.pop(_COUNTING_ID, None)
