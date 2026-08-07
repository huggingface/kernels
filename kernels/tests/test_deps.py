from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from kernels.deps import LocalKernel, RemoteKernel
from kernels.variants import parse_variant


def _variant():
    # A noarch variant so the test does not depend on a specific backend.
    return parse_variant("torch-cpu")


def test_local_kernel_eq_and_hash():
    a = LocalKernel(variant_path=Path("/tmp/a"))
    b = LocalKernel(variant_path=Path("/tmp/a"))
    c = LocalKernel(variant_path=Path("/tmp/b"))

    assert a == b
    assert hash(a) == hash(b)
    assert a != c

    assert {a, b, c} == {a, c}
    assert {a: 1}[b] == 1


def test_remote_kernel_eq_and_hash():
    v = _variant()
    a = RemoteKernel(repo_id="foo/bar", revision="deadbeef", variant=v)
    b = RemoteKernel(repo_id="foo/bar", revision="deadbeef", variant=v)
    c = RemoteKernel(repo_id="foo/bar", revision="cafef00d", variant=v)
    d = RemoteKernel(repo_id="other/repo", revision="deadbeef", variant=v)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != d

    assert {a, b, c, d} == {a, c, d}
    assert {a: 1}[b] == 1


def test_local_kernel_is_immutable():
    kernel = LocalKernel(variant_path=Path("/tmp/a"))
    with pytest.raises(FrozenInstanceError):
        kernel.variant_path = Path("/tmp/b")


def test_remote_kernel_is_immutable():
    v = _variant()
    kernel = RemoteKernel(repo_id="foo/bar", revision="deadbeef", variant=v)
    with pytest.raises(FrozenInstanceError):
        kernel.repo_id = "other/repo"
    with pytest.raises(FrozenInstanceError):
        kernel.revision = "cafef00d"
    with pytest.raises(FrozenInstanceError):
        kernel.variant = parse_variant("torch-cuda")
