from pathlib import Path

import pytest

from kernels_data import (
    KernelDependency,
    KernelLock,
    KernelLocks,
    KernelPaths,
    KernelVersion,
    NixKernelLock,
    NixKernelLocks,
)

COMMIT_RELU = "d649efb56fb249ac8f7a57fa1866728ad0c60e52"
COMMIT_VERSIONS = "f609e51b856b3d874b0ae8445913e200f02c1735"
SRI_HASH = "sha256-1CM0MGEOCqqnYV989jcUANEbiB727JftjXTdQVUHPwk="


def _dep(repo_id, version):
    return KernelDependency(repo_id=repo_id, version=KernelVersion.Version(version))


def _relu_dep():
    return _dep("kernels-community/relu", 1)


def _versions_dep():
    return _dep("kernels-test/versions", 2)


def _sample_locks():
    return KernelLocks(
        {
            _versions_dep(): KernelLock(COMMIT_VERSIONS),
            _relu_dep(): KernelLock(COMMIT_RELU),
        }
    )


def _sample_nix_locks():
    return NixKernelLocks(
        {
            _versions_dep(): NixKernelLock(COMMIT_VERSIONS, SRI_HASH),
            _relu_dep(): NixKernelLock(COMMIT_RELU, SRI_HASH),
        }
    )


def test_kernel_lock_commit():
    lock = KernelLock(COMMIT_RELU)
    assert lock.commit == COMMIT_RELU


def test_kernel_lock_invalid_commit():
    with pytest.raises(ValueError):
        KernelLock("d649efb")


def test_kernel_lock_json_round_trip():
    lock = KernelLock(COMMIT_RELU)
    assert KernelLock.from_json(lock.to_json()) == lock

    with pytest.raises(ValueError):
        KernelLock.from_json("{not json")


def test_kernel_locks_mapping_protocol():
    locks = _sample_locks()

    assert len(locks) == 2
    assert locks[_relu_dep()] == KernelLock(COMMIT_RELU)
    assert _versions_dep() in locks
    assert _dep("kernels-community/absent", 1) not in locks
    assert "not-a-dependency" not in locks
    assert set(iter(locks)) == {_relu_dep(), _versions_dep()}


def test_kernel_locks_keys_values_items():
    locks = _sample_locks()

    assert locks.keys() == [_relu_dep(), _versions_dep()]
    assert locks.values() == [KernelLock(COMMIT_RELU), KernelLock(COMMIT_VERSIONS)]
    assert locks.items() == [
        (_relu_dep(), KernelLock(COMMIT_RELU)),
        (_versions_dep(), KernelLock(COMMIT_VERSIONS)),
    ]


def test_kernel_locks_get_default():
    locks = _sample_locks()
    missing = _dep("kernels-community/absent", 1)
    default = KernelLock(COMMIT_VERSIONS)

    assert locks.get(_relu_dep()) == KernelLock(COMMIT_RELU)
    assert locks.get(missing) is None
    assert locks.get(missing, default) == default


def test_kernel_locks_getitem_missing_raises_key_error():
    with pytest.raises(KeyError):
        _sample_locks()[_dep("kernels-community/absent", 1)]


def test_kernel_locks_iteration_order_is_sorted():
    forward = KernelLocks(
        {
            _relu_dep(): KernelLock(COMMIT_RELU),
            _versions_dep(): KernelLock(COMMIT_VERSIONS),
        }
    )
    reverse = KernelLocks(
        {
            _versions_dep(): KernelLock(COMMIT_VERSIONS),
            _relu_dep(): KernelLock(COMMIT_RELU),
        }
    )

    assert forward.keys() == [_relu_dep(), _versions_dep()]
    assert forward.keys() == reverse.keys()


def test_kernel_locks_json_round_trip():
    locks = _sample_locks()
    assert KernelLocks.from_json(locks.to_json()) == locks


def test_kernel_locks_from_json_rejects_invalid():
    with pytest.raises(ValueError):
        KernelLocks.from_json("{not json")

    duplicate = f"""[
        {{"dependency": {{"repo-id": "kernels-test/versions", "version": 2}},
         "lock": {{"commit": "{COMMIT_VERSIONS}"}}}},
        {{"dependency": {{"repo-id": "kernels-test/versions", "version": 2}},
         "lock": {{"commit": "{COMMIT_RELU}"}}}}
    ]"""
    with pytest.raises(ValueError):
        KernelLocks.from_json(duplicate)


def test_nix_kernel_lock_construction():
    lock = NixKernelLock(COMMIT_RELU, SRI_HASH)
    assert lock.commit == COMMIT_RELU
    assert lock.hash == SRI_HASH

    with pytest.raises(ValueError):
        NixKernelLock("d649efb", SRI_HASH)


def test_nix_kernel_lock_json_round_trip():
    lock = NixKernelLock(COMMIT_RELU, SRI_HASH)
    assert NixKernelLock.from_json(lock.to_json()) == lock


def test_nix_kernel_locks_mapping_smoke():
    locks = _sample_nix_locks()

    assert len(locks) == 2
    assert locks[_relu_dep()] == NixKernelLock(COMMIT_RELU, SRI_HASH)
    assert _versions_dep() in locks
    assert locks.get(_dep("kernels-community/absent", 1)) is None


def test_nix_kernel_locks_json_round_trip():
    locks = _sample_nix_locks()
    assert NixKernelLocks.from_json(locks.to_json()) == locks


def test_nix_kernel_locks_entry_without_hash_is_rejected():
    json = (
        '[{"dependency": {"repo-id": "kernels-test/versions", "version": 2},'
        f' "lock": {{"commit": "{COMMIT_VERSIONS}"}}}}]'
    )
    with pytest.raises(ValueError):
        NixKernelLocks.from_json(json)


def test_kernel_paths_mapping_smoke():
    paths = KernelPaths(
        {
            _versions_dep(): "/kernels/versions",
            _relu_dep(): Path("/kernels/relu"),
        }
    )

    assert len(paths) == 2
    assert paths[_relu_dep()] == Path("/kernels/relu")
    assert paths[_versions_dep()] == Path("/kernels/versions")
    assert _versions_dep() in paths
    assert paths.get(_dep("kernels-community/absent", 1)) is None
    assert paths.values() == [Path("/kernels/relu"), Path("/kernels/versions")]


def test_kernel_paths_json_round_trip():
    paths = KernelPaths(
        {
            _versions_dep(): Path("/kernels/versions"),
            _relu_dep(): Path("/kernels/relu"),
        }
    )
    assert KernelPaths.from_json(paths.to_json()) == paths
