import pytest
from huggingface_hub.file_download import repo_folder_name

import kernels.hf_hub as hf_hub
from kernels._rust import KernelVersion, Oid
from kernels._versions import _resolve_ref

REPO_ID = "kernels-test/signatures"
COMMIT = "d649efb56fb249ac8f7a57fa1866728ad0c60e52"


@pytest.fixture
def cached_refs(tmp_path, monkeypatch):
    """A cache containing a single ref, so offline resolution is hermetic."""
    monkeypatch.setattr(hf_hub, "CACHE_DIR", str(tmp_path))
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
    import kernels._versions as versions

    def fail(*args, **kwargs):
        raise AssertionError("a version was resolved as if it were a ref")

    monkeypatch.setattr(versions, "_resolve_ref", fail)

    revision = versions.resolve_kernel_version(
        "kernels-community/relu",
        KernelVersion.Version(1),
        local_files_only=False,
    )

    assert revision == Oid.from_str(str(revision))
