import io
import tarfile
from pathlib import Path

import pytest

import kernels.jit as jit_module
from kernels.jit import (
    FallbackResolver,
    JITBuildError,
    JITBuildInfoError,
    JITOptions,
    _cache_key,
    _commit_from_source_url,
    _normalize_source,
    _reject_unsupported_free_threaded_python,
    _resolve_source_commit,
    _safe_extract,
    _source_fingerprint,
    _source_subdir,
    _SourceSpec,
)


class _MissingResolver:
    def resolve(self, *, api, backend, kernel):
        raise FileNotFoundError("no compatible build")


class _ResultResolver:
    def __init__(self, result):
        self.result = result

    def resolve(self, *, api, backend, kernel):
        return self.result


def test_fallback_resolver_uses_fallback_only_for_missing_build():
    fallback = _ResultResolver("fallback")
    resolver = FallbackResolver(primary=_MissingResolver(), fallback=fallback)
    assert resolver.resolve(api=None, backend="cpu", kernel=None) == "fallback"


def test_fallback_resolver_propagates_other_errors():
    class _BrokenResolver:
        def resolve(self, *, api, backend, kernel):
            raise ValueError("broken metadata")

    resolver = FallbackResolver(primary=_BrokenResolver(), fallback=_ResultResolver("fallback"))
    with pytest.raises(ValueError, match="broken metadata"):
        resolver.resolve(api=None, backend="cpu", kernel=None)


def test_normalize_github_tree_source():
    commit = "a" * 40
    source, subdir = _normalize_source(
        f"https://github.com/huggingface/kernels-community/tree/{commit}/quantization-gptq",
        None,
    )
    assert source == "https://github.com/huggingface/kernels-community"
    assert subdir == "quantization-gptq"
    assert (
        _commit_from_source_url(f"https://github.com/huggingface/kernels-community/tree/{commit}/quantization-gptq")
        == commit
    )


def test_normalize_local_source(tmp_path):
    source, subdir = _normalize_source(tmp_path, "kernel")
    assert source == tmp_path.resolve()
    assert subdir == "kernel"


def test_source_commit_requires_full_sha():
    assert _resolve_source_commit("a" * 40, None) == "a" * 40
    with pytest.raises(JITBuildError, match="40-character"):
        _resolve_source_commit("main", None)


def test_source_subdir_rejects_traversal(tmp_path):
    with pytest.raises(JITBuildError, match="escapes"):
        _source_subdir(tmp_path, "../outside")


def test_safe_extract_rejects_traversal(tmp_path):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        data = b"escape"
        info = tarfile.TarInfo("../escape")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    payload.seek(0)

    with tarfile.open(fileobj=payload, mode="r:gz") as archive:
        with pytest.raises(JITBuildError, match="Unsafe path"):
            _safe_extract(archive, tmp_path)


def test_local_source_fingerprint_changes_with_source(tmp_path):
    source_file = tmp_path / "build.toml"
    source_file.write_text("version = 1\n")
    source = _SourceSpec(source=tmp_path, subdir=None, commit=None)
    first = _source_fingerprint(source)
    source_file.write_text("version = 2\n")
    assert _source_fingerprint(source) != first


def test_cache_key_is_stable_and_tracks_source(tmp_path):
    source_file = tmp_path / "build.toml"
    source_file.write_text("version = 1\n")
    source = _SourceSpec(source=tmp_path, subdir=None, commit=None)

    class _Revision:
        def __str__(self):
            return "b" * 40

    revision = _Revision()
    first = _cache_key(repo_id="kernels-community/example", revision=revision, source=source)
    assert _cache_key(repo_id="kernels-community/example", revision=revision, source=source) == first

    source_file.write_text("version = 2\n")
    assert _cache_key(repo_id="kernels-community/example", revision=revision, source=source) != first


def test_jit_options_are_explicit():
    assert JITOptions().source is None
    assert JITOptions(source=Path("/tmp/source")).source == Path("/tmp/source")


def test_free_threaded_python_before_315_is_rejected(monkeypatch):
    monkeypatch.setattr(
        jit_module.sysconfig,
        "get_config_var",
        lambda name: 1 if name == "Py_GIL_DISABLED" else None,
    )
    monkeypatch.setattr(jit_module.sys, "version_info", (3, 14, 0))
    with pytest.raises(JITBuildInfoError, match="free-threaded"):
        _reject_unsupported_free_threaded_python()


def test_free_threaded_python_315_is_not_rejected(monkeypatch):
    monkeypatch.setattr(
        jit_module.sysconfig,
        "get_config_var",
        lambda name: 1 if name == "Py_GIL_DISABLED" else None,
    )
    monkeypatch.setattr(jit_module.sys, "version_info", (3, 15, 0))
    _reject_unsupported_free_threaded_python()
