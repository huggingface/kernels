from pathlib import Path

import pytest
from huggingface_hub.file_download import repo_folder_name

import kernels._versions as versions
import kernels.hf_hub as hf_hub
from kernels import install_kernel
from kernels._rust import KernelVersion, Oid
from kernels._versions import _resolve_ref

REPO_ID = "kernels-test/signatures"
COMMIT = "d649efb56fb249ac8f7a57fa1866728ad0c60e52"


@pytest.fixture
def cached_refs(tmp_path, monkeypatch):
    """A cache containing a single ref, so offline resolution is hermetic."""
    monkeypatch.setenv("KERNELS_CACHE", str(tmp_path))
    refs = tmp_path / repo_folder_name(repo_id=REPO_ID, repo_type="kernel") / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(COMMIT)
    return refs


def test_commit_resolves_without_contacting_the_hub(monkeypatch):
    """A revision that is already a commit must not cost a request.

    Lock files and `version=` both produce commits, so this is the common
    path and it has to stay free.
    """

    def fail():
        raise AssertionError("the Hub was contacted to resolve a commit")

    monkeypatch.setattr(hf_hub, "_get_hf_api", fail)

    assert _resolve_ref(REPO_ID, COMMIT, local_files_only=False) == Oid.from_str(COMMIT)


def test_commit_is_canonicalized(monkeypatch):
    def fail():
        raise AssertionError("the Hub was contacted to resolve a commit")

    monkeypatch.setattr(hf_hub, "_get_hf_api", fail)

    assert _resolve_ref(REPO_ID, COMMIT.upper(), local_files_only=False) == Oid.from_str(COMMIT)


def test_offline_resolution_uses_cached_refs(cached_refs):
    assert _resolve_ref(REPO_ID, "main", local_files_only=True) == Oid.from_str(COMMIT)


def test_offline_resolution_reports_an_uncached_ref(cached_refs):
    with pytest.raises(ValueError, match="not in the local cache"):
        _resolve_ref(REPO_ID, "some-branch", local_files_only=True)


def test_offline_resolution_stays_inside_the_refs_directory(cached_refs, tmp_path):
    """A ref is used as a path segment, so it must not be able to escape."""
    (tmp_path / "elsewhere").write_text(COMMIT)

    with pytest.raises(ValueError, match="not in the local cache"):
        _resolve_ref(REPO_ID, "../../../elsewhere", local_files_only=True)


def test_a_cached_ref_must_still_be_a_commit(cached_refs):
    (cached_refs / "main").write_text("not-a-commit")

    with pytest.raises(ValueError, match="Invalid git object id"):
        _resolve_ref(REPO_ID, "main", local_files_only=True)


@pytest.mark.parametrize("invalid", ["d649efb", "", "main", "z" * 40, COMMIT + "0"])
def test_oid_rejects_anything_but_a_full_object_id(invalid):
    with pytest.raises(ValueError, match="Invalid git object id"):
        Oid.from_str(invalid)


def test_a_version_needs_no_ref_resolution(monkeypatch):
    """Looking up a version already yields its commit.

    Guards against reintroducing a resolution step that would suggest a
    version can name something other than a commit.
    """

    def fail(*args, **kwargs):
        raise AssertionError("a version was resolved as if it were a ref")

    monkeypatch.setattr(versions, "_resolve_ref", fail)

    revision = versions.resolve_kernel_version(
        "kernels-community/relu",
        KernelVersion.Version(1),
        local_files_only=False,
    )

    assert revision == Oid.from_str(str(revision))


# Kernels are fetched by commit, so snapshot downloads do not create a ref.
# This is not a problem during normal usage, but breaks offline use if the
# kernel is resolved by version or non-commit ref. For this reason, we create
# a ref when resolving a name. The tests below ensure that this behavior is
# correct.


def _refs_of(cache_dir: Path, repo_id: str = REPO_ID) -> list[str]:
    refs = cache_dir / repo_folder_name(repo_id=repo_id, repo_type="kernel") / "refs"
    return sorted(p.name for p in refs.iterdir()) if refs.is_dir() else []


@pytest.mark.parametrize("version", [KernelVersion.Revision("v1"), KernelVersion.Version(1)])
def test_resolution_records_the_ref(tmp_path, monkeypatch, version):
    monkeypatch.setenv("KERNELS_CACHE", str(tmp_path))

    commit = versions.resolve_kernel_version("kernels-community/relu", version, local_files_only=False)

    assert _refs_of(tmp_path, "kernels-community/relu") == ["v1"]

    # Written verbatim, since huggingface_hub does not strip the file.
    ref_path = tmp_path / repo_folder_name(repo_id="kernels-community/relu", repo_type="kernel") / "refs" / "v1"
    assert ref_path.read_text() == str(commit)


def test_resolving_a_commit_records_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELS_CACHE", str(tmp_path))

    versions.resolve_kernel_version(REPO_ID, KernelVersion.Revision(COMMIT), local_files_only=False)

    assert _refs_of(tmp_path) == []


def test_recording_a_ref_tolerates_an_unwritable_cache(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("KERNELS_CACHE", str(tmp_path / "not-a-directory"))
    (tmp_path / "not-a-directory").write_text("")

    with caplog.at_level("WARNING", logger="kernels._versions"):
        versions._record_ref_in_cache(REPO_ID, "v1", COMMIT)

    assert "Could not record revision" in caplog.text


def test_recording_a_ref_stays_inside_the_refs_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELS_CACHE", str(tmp_path))

    versions._record_ref_in_cache(REPO_ID, "../../../escaped", COMMIT)

    assert not (tmp_path.parent.parent.parent / "escaped").exists()
    assert not (tmp_path / "escaped").exists()


def test_a_downloaded_revision_resolves_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELS_CACHE", str(tmp_path))

    expected = install_kernel("kernels-community/relu", revision="v1", backend="cpu")

    path = install_kernel("kernels-community/relu", revision="v1", backend="cpu", local_files_only=True)

    assert path == expected


def test_a_downloaded_version_resolves_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELS_CACHE", str(tmp_path))

    expected = install_kernel("kernels-community/relu", version=1, backend="cpu")

    path = install_kernel("kernels-community/relu", version=1, backend="cpu", local_files_only=True)

    assert path == expected
