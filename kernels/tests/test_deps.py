import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from kernels_data import KernelDependency, KernelVersion, Metadata

from kernels.deps import resolve_kernel_tree
from kernels.resolver import LocalKernel


def _dep(repo_id: str = "test/kernel", version: int = 1) -> KernelDependency:
    return KernelDependency(repo_id=repo_id, version=KernelVersion.Version(version))


def _metadata(*deps: KernelDependency) -> Metadata:
    return Metadata.from_bytes(
        json.dumps(
            {
                "id": "test_1_cpu",
                "name": "test",
                "version": 1,
                "license": "Apache-2.0",
                "python-depends": ["torch"],
                "kernel-depends": [{"repo-id": dep.repo_id, "version": dep.version.version} for dep in deps],
                "backend": {"type": "cpu"},
            }
        ).encode()
    )


def _local_kernel(repo_id: str, *deps: KernelDependency) -> LocalKernel:
    return LocalKernel(variant_path=Path("/repo") / repo_id, metadata=_metadata(*deps))


@dataclass
class _MapResolver:
    """Test resolver that resolves kernels from a repo ID mapping."""

    kernels: dict[str, LocalKernel]

    def resolve(self, *, api, backend, kernel) -> LocalKernel | None:
        return self.kernels.get(kernel.repo_id)


def test_resolve_kernel_tree_no_dependencies():
    kernel = _local_kernel("test/kernel")
    resolver = _MapResolver({"test/kernel": kernel})

    tree = resolve_kernel_tree(api=None, backend=None, kernel=_dep(), resolver=resolver)

    assert tree.location is kernel
    assert tree.deps == {}


def test_resolve_kernel_tree_builds_tree():
    a, b, c = _dep("test/a"), _dep("test/b"), _dep("test/c")
    kernel_a = _local_kernel("test/a", b)
    kernel_b = _local_kernel("test/b", c)
    kernel_c = _local_kernel("test/c")
    resolver = _MapResolver({"test/a": kernel_a, "test/b": kernel_b, "test/c": kernel_c})

    tree = resolve_kernel_tree(api=None, backend=None, kernel=a, resolver=resolver)

    assert tree.location is kernel_a
    assert set(tree.deps) == {"test/b"}
    node_b = tree.deps["test/b"]
    assert node_b.location is kernel_b
    assert set(node_b.deps) == {"test/c"}
    node_c = node_b.deps["test/c"]
    assert node_c.location is kernel_c
    assert node_c.deps == {}


@pytest.mark.parametrize(
    "kernel, message",
    [
        (_dep("test/kernel", version=1), "version: 1"),
        (KernelDependency(repo_id="test/kernel", version=KernelVersion.Revision("main")), "revision: main"),
    ],
)
@pytest.mark.parametrize("resolver", [None, _MapResolver({})])
def test_resolve_kernel_tree_unresolvable(kernel, resolver, message):
    with pytest.raises(ValueError, match="Could not resolve kernel") as exc_info:
        resolve_kernel_tree(api=None, backend=None, kernel=kernel, resolver=resolver)

    assert "test/kernel" in str(exc_info.value)
    assert message in str(exc_info.value)


def test_resolve_kernel_tree_detects_cycle():
    a, b = _dep("test/a"), _dep("test/b")
    resolver = _MapResolver(
        {
            "test/a": _local_kernel("test/a", b),
            "test/b": _local_kernel("test/b", a),
        }
    )

    with pytest.raises(ValueError, match="Cyclic kernel dependency detected: test/a"):
        resolve_kernel_tree(api=None, backend=None, kernel=a, resolver=resolver)


def test_resolve_kernel_tree_diamond_is_not_a_cycle():
    a, b, c, d = (_dep(f"test/{name}") for name in "abcd")
    kernel_d = _local_kernel("test/d")
    resolver = _MapResolver(
        {
            "test/a": _local_kernel("test/a", b, c),
            "test/b": _local_kernel("test/b", d),
            "test/c": _local_kernel("test/c", d),
            "test/d": kernel_d,
        }
    )

    tree = resolve_kernel_tree(api=None, backend=None, kernel=a, resolver=resolver)

    assert set(tree.deps) == {"test/b", "test/c"}
    assert tree.deps["test/b"].deps["test/d"].location is kernel_d
    assert tree.deps["test/c"].deps["test/d"].location is kernel_d
