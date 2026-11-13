import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.hf_api import GitRefInfo, HfApi

import kernels.backends as backends
import kernels.hf_hub as hf_hub
import kernels.importer as importer
import kernels.layer.func as func_module
import kernels.layer.layer as layer_module
import kernels.load as load
import kernels.resolver as resolver
from kernels import LockedFuncRepository, LockedLayerRepository
from kernels._rust import Digest, DigestAlgorithm, KernelDependency, KernelLock, KernelLocks, KernelVersion
from kernels._versions import _cached_refs_dir
from kernels.variants import parse_variant

REPO = "test/latest"
COMMITS = {1: "1" * 40, 2: "2" * 40, 10: "a" * 40}
MAIN = "b" * 40


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """Fake Hub transport with real version resolution, validation, and imports."""
    monkeypatch.delenv("UNSTABLE_KERNELS_ALL_LATEST", raising=False)
    monkeypatch.delenv("LOCAL_KERNELS", raising=False)
    monkeypatch.setenv("KERNELS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(load.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(backends, "_backend", lambda: backends.CPU())
    monkeypatch.setattr(importer, "_loaded_kernels", {})
    monkeypatch.setattr(importer, "_loaded_kernel_metadata", {})
    api = Mock(spec=HfApi)
    api.get_organization_overview.return_value = SimpleNamespace(trustedKernelPublisher=True)
    api.list_repo_refs.return_value = SimpleNamespace(
        branches=[GitRefInfo(name=f"v{v}", ref=f"refs/heads/v{v}", target_commit=c) for v, c in COMMITS.items()]
    )
    api.repo_info.return_value = SimpleNamespace(sha=MAIN)
    monkeypatch.setattr(hf_hub, "_get_hf_api", lambda **kwargs: api)
    monkeypatch.setattr(load, "_get_hf_api", lambda **kwargs: api)
    monkeypatch.setattr(layer_module, "_get_hf_api", lambda **kwargs: api)
    monkeypatch.setattr(func_module, "_get_hf_api", lambda **kwargs: api)
    monkeypatch.setattr(resolver, "get_variants", Mock(return_value=[parse_variant("torch-cpu")]))

    def snapshot(repo_id, *, revision, **kwargs):
        path = tmp_path / repo_id / revision
        if not path.exists():
            raise LocalEntryNotFoundError("No cached snapshot")
        return str(path)

    api.snapshot_download.side_effect = snapshot
    api.hf_hub_download.side_effect = lambda repo_id, filename, **kwargs: str(
        tmp_path / repo_id / kwargs["revision"] / filename
    )

    def write_kernel(repo_id=REPO, version=10, deps=()):
        commit = COMMITS.get(version, MAIN)
        variant = tmp_path / repo_id / commit / "build" / "torch-cpu"
        variant.mkdir(parents=True, exist_ok=True)
        (variant / "__init__.py").write_text(
            "from kernels import get_kernel_dep\n"
            "from types import SimpleNamespace\n"
            "from torch import nn\n"
            f"version = {version}\n"
            f"dependencies = {{repo: get_kernel_dep(repo) for repo in {list(deps)!r}}}\n"
            "def get_version():\n"
            "    return version\n"
            "class Version(nn.Module):\n"
            "    def forward(self):\n"
            "        return version\n"
            "layers = SimpleNamespace(Version=Version)\n"
        )
        (variant / "metadata.json").write_text(
            json.dumps(
                {
                    "id": f"latest_test_{repo_id.replace('/', '_')}_{version}",
                    "name": "latest-test",
                    "version": version,
                    "license": "Apache-2.0",
                    "python-depends": [],
                    "kernel-depends": [{"repo-id": repo, "version": 1} for repo in deps],
                    "backend": {"type": "cpu"},
                    "digest": {
                        "algorithm": "sha256",
                        "files": Digest.hash_variant(DigestAlgorithm.SHA256, variant).files,
                    },
                }
            )
        )
        return variant

    for version in [1, 2, 10, 0]:
        write_kernel(version=version)

    dep = KernelDependency(repo_id=REPO, version=KernelVersion.Version(1))
    locks = KernelLocks({dep: KernelLock(COMMITS[1])})
    lockfile = tmp_path / "kernels.lock"
    lockfile.write_text(locks.to_json())

    yield SimpleNamespace(api=api, write_kernel=write_kernel, lockfile=lockfile, locks=locks, dep=dep)

    for name in list(sys.modules):
        if name.startswith("latest_test_"):
            sys.modules.pop(name)


@pytest.mark.parametrize("loader", [load.get_kernel, load.has_kernel])
@pytest.mark.parametrize(
    "requested", [{"version": 1}, {"version": 99}, {"revision": "missing"}, {"revision": COMMITS[1]}, {}]
)
def test_latest_overrides_requested_version(hub, monkeypatch, caplog, loader, requested):
    monkeypatch.setenv("UNSTABLE_KERNELS_ALL_LATEST", "1")

    result = loader(REPO, backend="cpu", **requested)

    if loader is load.has_kernel:
        assert result is True
    else:
        assert result.version == 10
    resolver.get_variants.assert_called_once_with(hub.api, repo_id=REPO, revision=COMMITS[10])
    assert "UNSTABLE_KERNELS_ALL_LATEST" in caplog.text
    assert REPO in caplog.text
    assert f"v10 ({COMMITS[10]})" in caplog.text
    assert "may break your code" in caplog.text
    assert (_cached_refs_dir(REPO) / "v10").read_text() == COMMITS[10]


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "typo"])
def test_latest_is_opt_in(hub, monkeypatch, value, caplog):
    if value is not None:
        monkeypatch.setenv("UNSTABLE_KERNELS_ALL_LATEST", value)

    assert load.get_kernel(REPO, version=1, backend="cpu").version == 1
    assert load.get_kernel(REPO, revision=COMMITS[2], backend="cpu").version == 2
    for loader in [load.get_kernel, load.has_kernel]:
        with pytest.raises(ValueError, match="A kernel version or revision must be specified"):
            loader(REPO)
    assert "UNSTABLE_KERNELS_ALL_LATEST" not in caplog.text


@pytest.mark.parametrize("value", ["1", "true", "True", "YES", "on"])
def test_latest_setting_is_read_on_each_call(hub, monkeypatch, value):
    assert load.get_kernel(REPO, version=1, backend="cpu").version == 1
    monkeypatch.setenv("UNSTABLE_KERNELS_ALL_LATEST", value)
    assert load.get_kernel(REPO, version=1, backend="cpu").version == 10
    monkeypatch.delenv("UNSTABLE_KERNELS_ALL_LATEST")
    assert load.get_kernel(REPO, version=1, backend="cpu").version == 1


@pytest.mark.parametrize("loader", [load.get_kernel, load.has_kernel])
def test_latest_rejects_version_and_revision_together(hub, monkeypatch, loader):
    monkeypatch.setenv("UNSTABLE_KERNELS_ALL_LATEST", "1")
    with pytest.raises(ValueError, match="Only one of"):
        loader(REPO, version=1, revision="main")
    hub.api.list_repo_refs.assert_not_called()


def test_latest_unversioned_repo_uses_main(hub, monkeypatch):
    monkeypatch.setenv("UNSTABLE_KERNELS_ALL_LATEST", "1")
    hub.api.list_repo_refs.return_value = SimpleNamespace(
        branches=[GitRefInfo(name="main", ref="refs/heads/main", target_commit=MAIN)]
    )

    assert load.get_kernel(REPO, version=1, backend="cpu").version == 0
    hub.api.repo_info.assert_called_once_with(repo_id=REPO, repo_type="kernel", revision="main")
    assert (_cached_refs_dir(REPO) / "main").read_text() == MAIN


def test_latest_preserves_publisher_trust(hub, monkeypatch):
    monkeypatch.setenv("UNSTABLE_KERNELS_ALL_LATEST", "1")
    hub.api.get_organization_overview.return_value = SimpleNamespace(trustedKernelPublisher=False)
    with pytest.raises(ValueError, match="not from a trusted publisher"):
        load.get_kernel(REPO, version=1, backend="cpu")
    hub.api.list_repo_refs.assert_not_called()

    assert load.get_kernel(REPO, version=1, backend="cpu", trust_remote_code=[REPO]).version == 10


def test_latest_does_not_fall_back_to_compatible_older_version(hub, monkeypatch):
    monkeypatch.setenv("UNSTABLE_KERNELS_ALL_LATEST", "1")
    resolver.get_variants.side_effect = lambda api, *, repo_id, revision: (
        [] if revision in {COMMITS[10], "refs/heads/v10"} else [parse_variant("torch-cpu")]
    )

    assert not load.has_kernel(REPO, version=1, backend="cpu")
    with pytest.raises(FileNotFoundError, match="Cannot find a build variant"):
        load.get_kernel(REPO, version=1, backend="cpu")
    hub.api.snapshot_download.assert_not_called()


def test_local_override_takes_precedence(hub, monkeypatch):
    monkeypatch.setenv("UNSTABLE_KERNELS_ALL_LATEST", "1")
    local_path = hub.write_kernel(version=1)
    monkeypatch.setenv("LOCAL_KERNELS", f"{REPO}={local_path}")

    assert load.get_kernel(REPO, backend="cpu").version == 1
    assert load.has_kernel(REPO, backend="cpu")
    assert hub.api.mock_calls == []


@pytest.mark.parametrize("local_root", [False, True])
def test_latest_applies_to_transitive_dependencies(hub, monkeypatch, local_root):
    monkeypatch.setenv("UNSTABLE_KERNELS_ALL_LATEST", "1")
    root = hub.write_kernel(deps=["test/child"])
    hub.write_kernel("test/child", deps=["test/grandchild"])
    hub.write_kernel("test/grandchild")

    kernel = (
        load.get_local_kernel(root, backend="cpu") if local_root else load.get_kernel(REPO, version=1, backend="cpu")
    )

    child = kernel.dependencies["test/child"]
    assert child.version == 10
    assert child.dependencies["test/grandchild"].version == 10


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("loader", [load.get_locked_kernel, load.load_kernel])
@pytest.mark.parametrize("enabled", [False, True])
def test_locked_loading(hub, monkeypatch, loader, offline, enabled):
    """
    Verifies that cache-only calls avoid network lookup/download methods
    and that the lockfile remains unchanged.
    """
    if enabled:
        monkeypatch.setenv("UNSTABLE_KERNELS_ALL_LATEST", "1")
    monkeypatch.setattr(load.constants, "HF_HUB_OFFLINE", offline)
    refs = _cached_refs_dir(REPO)
    refs.mkdir(parents=True)
    (refs / "v2").write_text(COMMITS[2])
    original_lockfile = hub.lockfile.read_text()

    kernel = loader(REPO, lockfile=hub.lockfile)

    cache_only = offline or loader is load.load_kernel
    expected_version = (2 if cache_only else 10) if enabled else 1
    assert kernel.version == expected_version
    assert hub.api.snapshot_download.call_args.kwargs["local_files_only"] is cache_only
    if cache_only:
        hub.api.list_repo_refs.assert_not_called()
        hub.api.repo_info.assert_not_called()
        hub.api.hf_hub_download.assert_not_called()
    assert hub.lockfile.read_text() == original_lockfile


def test_latest_locked_kernel_can_add_dependencies(hub, monkeypatch):
    monkeypatch.setenv("UNSTABLE_KERNELS_ALL_LATEST", "1")
    hub.write_kernel(deps=["test/new-dependency"])
    hub.write_kernel("test/new-dependency")
    monkeypatch.setattr(load, "get_caller_locked_kernel_revision", lambda repo_id: (hub.locks, hub.dep))

    kernel = load.get_locked_kernel(REPO, lockfile=None)

    assert kernel.version == 10
    assert kernel.dependencies["test/new-dependency"].version == 10


@pytest.mark.parametrize("repository", [LockedLayerRepository, LockedFuncRepository])
@pytest.mark.parametrize("offline", [False, True])
def test_latest_locked_layer_mapping(hub, monkeypatch, repository, offline):
    kwargs = {"layer_name": "Version"} if repository is LockedLayerRepository else {"func_name": "get_version"}
    repo = repository(REPO, lockfile=hub.lockfile, **kwargs)
    layer = repo.load()
    assert layer()() == 1

    monkeypatch.setenv("UNSTABLE_KERNELS_ALL_LATEST", "1")
    monkeypatch.setattr(load.constants, "HF_HUB_OFFLINE", offline)
    refs = _cached_refs_dir(REPO)
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "v2").write_text(COMMITS[2])

    layer = repo.load()
    assert layer()() == (2 if offline else 10)


@pytest.mark.parametrize("loader", [load.get_kernel, load.has_kernel])
@pytest.mark.parametrize("ref, commit, expected", [("v2", COMMITS[2], 2), ("main", MAIN, 0)])
def test_latest_offline_uses_only_cached_refs(hub, monkeypatch, loader, ref, commit, expected):
    monkeypatch.setenv("UNSTABLE_KERNELS_ALL_LATEST", "1")
    monkeypatch.setattr(load.constants, "HF_HUB_OFFLINE", True)
    refs = _cached_refs_dir(REPO)
    refs.mkdir(parents=True)
    (refs / ref).write_text(commit)

    result = loader(REPO, version=1, backend="cpu")

    if loader is load.has_kernel:
        assert result is True
    else:
        assert result.version == expected
    hub.api.list_repo_refs.assert_not_called()
    hub.api.repo_info.assert_not_called()
    hub.api.hf_hub_download.assert_not_called()
    assert hub.api.snapshot_download.call_args.kwargs["local_files_only"] is True


def test_latest_offline_missing_snapshot_does_not_use_pinned_version(hub, monkeypatch):
    monkeypatch.setenv("UNSTABLE_KERNELS_ALL_LATEST", "1")
    monkeypatch.setattr(load.constants, "HF_HUB_OFFLINE", True)
    refs = _cached_refs_dir(REPO)
    refs.mkdir(parents=True)
    (refs / "v1").write_text(COMMITS[1])
    (refs / "v20").write_text("c" * 40)

    assert not load.has_kernel(REPO, version=1, backend="cpu")
    with pytest.raises(FileNotFoundError, match="Cannot find a local snapshot"):
        load.get_kernel(REPO, version=1, backend="cpu")
    hub.api.list_repo_refs.assert_not_called()
