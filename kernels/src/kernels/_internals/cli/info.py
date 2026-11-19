import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import constants
from huggingface_hub.errors import HfHubHTTPError

from kernels._internals._versions import _get_available_versions, revision_or_version
from kernels._internals.deps import DepTreeNode, resolve_kernel_tree
from kernels._internals.hf_hub import _get_hf_api
from kernels._internals.resolver import (
    HubCacheResolver,
    HubResolver,
    LocalKernel,
    RemoteKernel,
    RepoPathsResolver,
    Resolver,
    SequentialResolver,
)
from kernels._internals.variants import (
    Noarch,
    Variant,
    get_variants,
    get_variants_local,
)
from kernels._rust import KernelDependency, KernelVersion, Metadata


def print_kernel_info(
    kernel: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    json_output: bool = False,
    dep_tree: bool = False,
):
    """Describe a kernel from a local path or a Hub repo ID."""
    path = Path(kernel)
    try:
        if revision is not None and version is not None:
            raise ValueError("Only one of --revision or --version can be specified")

        # Only inspect metadata.
        resolver: Resolver = (
            HubCacheResolver(trust_remote_code=True)
            if constants.HF_HUB_OFFLINE
            else HubResolver(trust_remote_code=True)
        )
        if path.is_dir():
            if revision is not None or version is not None:
                raise ValueError("--revision and --version cannot be used with a local path")
            resolver = SequentialResolver([RepoPathsResolver(local_kernels={str(path): path}), resolver])
            dependency = KernelDependency(repo_id=str(path), version=KernelVersion.Version(0))
            source = {"path": str(path)}
        else:
            if revision is None and version is None:
                versions = _get_available_versions(kernel, local_files_only=constants.HF_HUB_OFFLINE)
                if versions:
                    version = max(versions)
                else:
                    revision = "main"
            dependency = KernelDependency(
                repo_id=kernel, version=revision_or_version(revision=revision, version=version)
            )
            source = {"repo_id": kernel, "revision": revision if revision is not None else f"v{version}"}

        info = _kernel_info(dependency, resolver=resolver, dep_tree=dep_tree)
        info.update(source)
    except (OSError, ValueError, HfHubHTTPError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    if json_output:
        print(json.dumps(info, indent=2))
    else:
        _print_human(info)


def _kernel_info(
    kernel: KernelDependency,
    *,
    resolver: Resolver,
    dep_tree: bool,
) -> dict:
    api = _get_hf_api()
    tree = resolve_kernel_tree(api=api, backend=None, kernel=kernel, resolver=resolver)
    location = tree.location
    if isinstance(location, LocalKernel):
        variants = get_variants_local(location.variant_path.parent)
    else:
        variants = get_variants(api, repo_id=location.repo_id, revision=str(location.revision))

    info = _metadata_info(location.metadata, variants)
    info["variant"] = location.variant_str
    info.update(_dependency_info(tree, dep_tree=dep_tree))
    return info


def _dependency_info(tree: DepTreeNode[LocalKernel | RemoteKernel], *, dep_tree: bool) -> dict:
    kernel_depends: set[str] = set()
    python_depends = set(tree.location.metadata.python_depends)

    def visit(nodes: dict[str, DepTreeNode]) -> dict:
        tree = {}
        for repo_id, node in sorted(nodes.items()):
            kernel_depends.add(repo_id)
            python_depends.update(node.location.metadata.python_depends)
            tree[repo_id] = visit(node.deps)
        return tree

    dependency_tree = visit(tree.deps)
    info: dict[str, Any] = {"kernel_depends": sorted(kernel_depends), "python_depends": sorted(python_depends)}
    if dep_tree:
        info["dependency_tree"] = dependency_tree
    return info


def _metadata_info(metadata: Metadata | None, variants: list[Variant]) -> dict:
    backends = set()
    for variant in variants:
        backend = variant.backend
        if backend is None:
            # Backend-agnostic SDK variant.
            continue
        if isinstance(backend, Noarch):
            backends.add(backend.backend_name)
        else:
            backends.add(backend.name)

    info: dict[str, Any] = {}
    if metadata is not None:
        info.update(
            {
                "name": str(metadata.name),
                "version": metadata.version,
                "kernels_minver": (str(metadata.kernels_minver) if metadata.kernels_minver is not None else None),
                "license": metadata.license,
                "upstream": metadata.upstream,
                "source": metadata.source,
                "python_depends": metadata.python_depends,
            }
        )

    info["backends"] = sorted(backends)

    return info


def _print_human(info: dict):
    def value(v) -> str:
        return "-" if v is None else str(v)

    if "repo_id" in info:
        print(f"Repository: {info['repo_id']}")
        print(f"Revision: {info['revision']}")
    else:
        print(f"Path: {info['path']}")

    print(f"Variant: {info['variant']}")
    print(f"Name: {value(info.get('name'))}")
    print(f"Version: {value(info.get('version'))}")
    print(f"Minimum kernels version: {value(info.get('kernels_minver'))}")
    print(f"License: {value(info.get('license'))}")
    print(f"Upstream: {value(info.get('upstream'))}")
    print(f"Source: {value(info.get('source'))}")
    if "dependency_tree" in info:
        print("Kernel dependencies:")
        print(info.get("repo_id", info.get("path")))
        _print_dependency_tree(info["dependency_tree"])
    else:
        kernel_depends = info["kernel_depends"]
        print(f"Kernel dependencies: {', '.join(kernel_depends) if kernel_depends else '-'}")
    python_depends = info.get("python_depends")
    print(f"Python dependencies: {', '.join(python_depends) if python_depends else '-'}")
    print(f"Backends: {', '.join(info['backends'])}")


def _print_dependency_tree(tree: dict, prefix: str = "") -> None:
    for index, (repo_id, deps) in enumerate(tree.items()):
        last = index == len(tree) - 1
        print(f"{prefix}{'`-- ' if last else '|-- '}{repo_id}")
        _print_dependency_tree(deps, prefix + ("    " if last else "|   "))
