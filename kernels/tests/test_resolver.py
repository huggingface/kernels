import json
import re
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest
from kernels_data import (
    KernelDependency,
    KernelLock,
    KernelLocks,
    KernelName,
    KernelPaths,
    KernelVersion,
    Metadata,
)

from kernels._versions import resolve_version_spec_as_ref
from kernels.hf_hub import _get_hf_api
from kernels.install import install_kernel
from kernels.resolver import (
    HubCacheResolver,
    HubResolver,
    KernelPathsResolver,
    LocalKernel,
    LockedHubCacheResolver,
    LockedHubResolver,
    NoopResolver,
    RemoteKernel,
    RepoPathsResolver,
    Resolver,
    SequentialResolver,
    _locked_revision,
)
from kernels.variants import parse_variant


@pytest.fixture(scope="module")
def api():
    return _get_hf_api()


@pytest.fixture(scope="module")
def installed_relu_cpu():
    """Install the relu kernel CPU variant, so that it is in the local cache."""
    return install_kernel("kernels-community/relu", revision="v1", backend="cpu")


@pytest.fixture(scope="module")
def relu_locks():
    """Locks for `kernels-community/relu` version 1, mirroring a lock file."""
    commit = resolve_version_spec_as_ref("kernels-community/relu", 1, local_files_only=False).target_commit
    dep = KernelDependency(repo_id="kernels-community/relu", version=KernelVersion.Version(1))
    return dep, KernelLocks({dep: KernelLock(commit=commit)}), commit


def _dep(repo_id: str = "test/kernel", version: int = 1) -> KernelDependency:
    return KernelDependency(repo_id=repo_id, version=KernelVersion.Version(version))


def _metadata() -> Metadata:
    return Metadata.from_bytes(
        json.dumps(
            {
                "id": "test_1_cpu",
                "name": "test",
                "version": 1,
                "license": "Apache-2.0",
                "python-depends": ["torch"],
                "backend": {"type": "cpu"},
            }
        ).encode()
    )


def _local_kernel(variant_path: str) -> LocalKernel:
    return LocalKernel(variant_path=Path(variant_path), metadata=_metadata())


def _variant():
    # A noarch variant so the test does not depend on a specific backend.
    return parse_variant("torch-cpu")


def _write_variant(repo_path: Path, variant: str = "torch-cpu") -> Path:
    """Write a fake kernel build variant (CPU noarch) under `repo_path/build`."""
    variant_dir = repo_path / "build" / variant
    variant_dir.mkdir(parents=True)
    (variant_dir / "metadata.json").write_text(
        json.dumps(
            {
                "id": "test_1_cpu",
                "name": "test",
                "version": 1,
                "license": "Apache-2.0",
                "python-depends": ["torch"],
                "backend": {"type": "cpu"},
            }
        )
    )
    return variant_dir


def test_local_kernel_eq_and_hash():
    m = _metadata()
    a = LocalKernel(variant_path=Path("/tmp/a"), metadata=m)
    b = LocalKernel(variant_path=Path("/tmp/a"), metadata=m)
    c = LocalKernel(variant_path=Path("/tmp/b"), metadata=m)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c

    assert {a, b, c} == {a, c}
    assert {a: 1}[b] == 1


def test_remote_kernel_eq_and_hash():
    v = _variant()
    m = _metadata()
    a = RemoteKernel(repo_id="foo/bar", revision="deadbeef", variant=v, metadata=m)
    b = RemoteKernel(repo_id="foo/bar", revision="deadbeef", variant=v, metadata=m)
    c = RemoteKernel(repo_id="foo/bar", revision="cafef00d", variant=v, metadata=m)
    d = RemoteKernel(repo_id="other/repo", revision="deadbeef", variant=v, metadata=m)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != d

    assert {a, b, c, d} == {a, c, d}
    assert {a: 1}[b] == 1


def test_local_kernel_is_immutable():
    kernel = LocalKernel(variant_path=Path("/tmp/a"), metadata=_metadata())
    with pytest.raises(FrozenInstanceError):
        kernel.variant_path = Path("/tmp/b")


def test_remote_kernel_is_immutable():
    v = _variant()
    kernel = RemoteKernel(repo_id="foo/bar", revision="deadbeef", variant=v, metadata=_metadata())
    with pytest.raises(FrozenInstanceError):
        kernel.repo_id = "other/repo"
    with pytest.raises(FrozenInstanceError):
        kernel.revision = "cafef00d"
    with pytest.raises(FrozenInstanceError):
        kernel.variant = parse_variant("torch-cuda")


def test_noop_resolver_never_resolves(api):
    assert NoopResolver().resolve(api=api, backend=None, kernel=_dep()) is None


@dataclass
class _FixedResolver:
    """Test resolver that always returns a fixed result."""

    result: LocalKernel | None

    def resolve(self, *, api, backend, kernel) -> LocalKernel | None:
        return self.result


def test_sequential_resolver_returns_first_match(api):
    first = _local_kernel("/tmp/first")
    second = _local_kernel("/tmp/second")
    resolver = SequentialResolver(resolvers=[_FixedResolver(first), _FixedResolver(second)])

    assert resolver.resolve(api=api, backend=None, kernel=_dep()) is first


def test_sequential_resolver_skips_non_matching(api):
    match = _local_kernel("/tmp/match")
    resolver = SequentialResolver(resolvers=[NoopResolver(), _FixedResolver(None), _FixedResolver(match)])

    assert resolver.resolve(api=api, backend=None, kernel=_dep()) is match


def test_sequential_resolver_returns_none_when_nothing_matches(api):
    resolver = SequentialResolver(resolvers=[NoopResolver(), _FixedResolver(None)])

    assert resolver.resolve(api=api, backend=None, kernel=_dep()) is None


@pytest.mark.parametrize("path_kind", ["repo", "build", "variant"])
def test_repo_paths_resolver_path_layouts(api, tmp_path, path_kind):
    variant_dir = _write_variant(tmp_path)
    repo_path = {
        "repo": tmp_path,
        "build": tmp_path / "build",
        "variant": variant_dir,
    }[path_kind]

    resolver = RepoPathsResolver(local_kernels={"test/kernel": repo_path})
    location = resolver.resolve(api=api, backend="cpu", kernel=_dep("test/kernel"))

    assert isinstance(location, LocalKernel)
    assert location.variant_path == variant_dir
    assert location.metadata.name == KernelName("test")


def test_repo_paths_resolver_unknown_repo(api, tmp_path):
    _write_variant(tmp_path)
    resolver = RepoPathsResolver(local_kernels={"test/kernel": tmp_path})

    assert resolver.resolve(api=api, backend="cpu", kernel=_dep("other/repo")) is None


def test_repo_paths_resolver_no_matching_variant(api, tmp_path):
    resolver = RepoPathsResolver(local_kernels={"test/kernel": tmp_path})

    with pytest.raises(FileNotFoundError, match="Cannot find a build variant"):
        resolver.resolve(api=api, backend="cpu", kernel=_dep("test/kernel"))


def test_kernel_paths_resolver_resolves_known_dep(api, tmp_path):
    variant_dir = _write_variant(tmp_path)
    dep = _dep("test/kernel", version=2)
    resolver = KernelPathsResolver(kernel_paths=KernelPaths({dep: tmp_path}))

    location = resolver.resolve(api=api, backend="cpu", kernel=dep)

    assert isinstance(location, LocalKernel)
    assert location.variant_path == variant_dir


def test_kernel_paths_resolver_unknown_dep(api, tmp_path):
    _write_variant(tmp_path)
    resolver = KernelPathsResolver(kernel_paths=KernelPaths({_dep("test/kernel", version=1): tmp_path}))
    assert resolver.resolve(api=api, backend="cpu", kernel=_dep("test/kernel", version=2)) is None


def test_hub_resolver_resolves_remote_kernel(api):
    location = HubResolver(trust_remote_code=False).resolve(
        api=api, backend="cpu", kernel=_dep("kernels-community/relu", version=1)
    )

    assert isinstance(location, RemoteKernel)
    assert location.repo_id == "kernels-community/relu"
    assert re.fullmatch(r"[0-9a-f]{40}", location.revision)
    assert location.metadata.name == KernelName("relu")


def test_hub_resolver_revision_passthrough(api):
    location = HubResolver(trust_remote_code=False).resolve(
        api=api,
        backend="cpu",
        kernel=KernelDependency(repo_id="kernels-community/relu", version=KernelVersion.Revision("v1")),
    )

    assert location.revision == "v1"


def test_hub_resolver_blocks_untrusted_org(api):
    with pytest.raises(ValueError, match="not from a trusted publisher"):
        HubResolver(trust_remote_code=False).resolve(
            api=api, backend="cpu", kernel=_dep("kernels-test-untrusted/not-a-trused-org-kernel", version=1)
        )


@pytest.mark.cuda_only
def test_hub_resolver_trust_remote_code_bypasses_check(api):
    location = HubResolver(trust_remote_code=True).resolve(
        api=api, backend="cuda", kernel=_dep("kernels-test-untrusted/ci-test-kernel", version=1)
    )

    assert isinstance(location, RemoteKernel)


def test_hub_resolver_no_matching_variant(api):
    with pytest.raises(FileNotFoundError, match="Cannot find a build variant"):
        HubResolver(trust_remote_code=False).resolve(
            api=api,
            backend="cpu",
            kernel=KernelDependency(repo_id="kernels-test/only-torch-2.4", version=KernelVersion.Revision("main")),
        )


def test_hub_cache_resolver_resolves_cached_kernel(api, installed_relu_cpu):
    location = HubCacheResolver(trust_remote_code=False).resolve(
        api=api, backend="cpu", kernel=_dep("kernels-community/relu", version=1)
    )

    assert isinstance(location, LocalKernel)
    assert location.variant_path == installed_relu_cpu


def test_hub_cache_resolver_revision_passthrough(api, installed_relu_cpu):
    location = HubCacheResolver(trust_remote_code=False).resolve(
        api=api,
        backend="cpu",
        kernel=KernelDependency(repo_id="kernels-community/relu", version=KernelVersion.Revision("v1")),
    )

    assert location.variant_path == installed_relu_cpu


def test_hub_cache_resolver_uncached_repo(api):
    with pytest.raises(FileNotFoundError, match="local snapshot"):
        HubCacheResolver(trust_remote_code=False).resolve(
            api=api,
            backend="cpu",
            kernel=KernelDependency(
                repo_id="kernels-test/this-repo-should-not-exist",
                version=KernelVersion.Revision("0" * 40),
            ),
        )


def test_locked_hub_resolver_resolves_locked_revision(api, relu_locks):
    dep, locks, commit = relu_locks

    location = LockedHubResolver(kernel_locks=locks, trust_remote_code=False).resolve(
        api=api, backend="cpu", kernel=dep
    )

    assert isinstance(location, RemoteKernel)
    assert location.revision == commit


def test_locked_hub_resolver_requires_lock(api, relu_locks):
    _, locks, _ = relu_locks

    with pytest.raises(ValueError, match="is not locked"):
        LockedHubResolver(kernel_locks=locks, trust_remote_code=False).resolve(
            api=api, backend="cpu", kernel=_dep("kernels-community/relu", version=2)
        )


def test_locked_hub_cache_resolver_resolves_locked_kernel(api, relu_locks, installed_relu_cpu):
    dep, locks, _ = relu_locks

    location = LockedHubCacheResolver(kernel_locks=locks, trust_remote_code=False).resolve(
        api=api, backend="cpu", kernel=dep
    )

    assert isinstance(location, LocalKernel)
    assert location.variant_path == installed_relu_cpu


def test_locked_hub_cache_resolver_requires_lock(api, relu_locks):
    _, locks, _ = relu_locks

    with pytest.raises(ValueError, match="is not locked"):
        LockedHubCacheResolver(kernel_locks=locks, trust_remote_code=False).resolve(
            api=api, backend="cpu", kernel=_dep("kernels-community/relu", version=2)
        )


def test_locked_revision_returns_commit():
    dep = _dep("test/kernel", version=1)
    locks = KernelLocks({dep: KernelLock(commit="a" * 40)})

    assert _locked_revision(locks, dep) == "a" * 40


def test_locked_revision_requires_lock():
    with pytest.raises(ValueError, match="is not locked"):
        _locked_revision(KernelLocks({}), _dep("test/kernel", version=1))


@pytest.mark.parametrize(
    "resolver",
    [
        NoopResolver(),
        SequentialResolver(resolvers=[]),
        RepoPathsResolver(local_kernels={}),
        KernelPathsResolver(kernel_paths=KernelPaths({})),
        HubResolver(trust_remote_code=False),
        HubCacheResolver(trust_remote_code=False),
        LockedHubResolver(kernel_locks=KernelLocks({}), trust_remote_code=False),
        LockedHubCacheResolver(kernel_locks=KernelLocks({}), trust_remote_code=False),
    ],
)
def test_resolver_protocol_conformance(resolver):
    assert isinstance(resolver, Resolver)
