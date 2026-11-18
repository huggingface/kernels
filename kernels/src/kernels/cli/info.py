import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import constants
from huggingface_hub.errors import HfHubHTTPError

from kernels._rust import Metadata
from kernels._versions import _get_available_versions, resolve_version_spec_as_ref
from kernels.deps import DepTreeNode, resolve_kernel_tree
from kernels.hf_hub import _get_cache_dir, _get_hf_api
from kernels.resolver import HubCacheResolver, HubResolver
from kernels.variants import (
    Noarch,
    Variant,
    get_variants,
    get_variants_local,
    resolve_variant,
    variants_trace_str,
)


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
        if path.is_dir():
            if revision is not None or version is not None:
                raise ValueError("--revision and --version cannot be used with a local path")
            info = _local_kernel_info(path, dep_tree=dep_tree)
        else:
            info = _hub_kernel_info(kernel, revision=revision, version=version, dep_tree=dep_tree)
    except (OSError, ValueError, HfHubHTTPError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    if json_output:
        print(json.dumps(info, indent=2))
    else:
        _print_human(info)


def _hub_kernel_info(
    repo_id: str,
    *,
    revision: str | None,
    version: int | None,
    dep_tree: bool,
) -> dict:
    if revision is not None and version is not None:
        print("Only one of --revision or --version can be specified", file=sys.stderr)
        sys.exit(1)

    if version is not None:
        revision = resolve_version_spec_as_ref(repo_id, version, local_files_only=constants.HF_HUB_OFFLINE).name
    elif revision is None:
        versions = _get_available_versions(repo_id, local_files_only=constants.HF_HUB_OFFLINE)
        revision = versions[max(versions.keys())].name if versions else "main"

    api = _get_hf_api()
    variants = get_variants(api, repo_id=repo_id, revision=revision)
    if not variants:
        print(
            f"No build variants found in {repo_id} (revision: {revision})",
            file=sys.stderr,
        )
        sys.exit(1)

    variant = _select_variant(variants, location=f"{repo_id} (revision: {revision})")
    metadata_path = api.hf_hub_download(
        repo_id,
        repo_type="kernel",
        filename=f"build/{variant.variant_str}/metadata.json",
        cache_dir=_get_cache_dir(),
        revision=revision,
        local_files_only=constants.HF_HUB_OFFLINE,
    )
    metadata = Metadata.read_from_file(metadata_path)

    info = {"repo_id": repo_id, "revision": revision, "variant": variant.variant_str}
    info.update(_metadata_info(metadata, variants))
    info.update(_dependency_info(metadata, dep_tree=dep_tree))
    return info


def _local_kernel_info(repo_path: Path, *, dep_tree: bool) -> dict:
    build_path = repo_path / "build"
    variants = get_variants_local(build_path)
    if not variants:
        print(f"No build variants found in: {build_path}", file=sys.stderr)
        sys.exit(1)

    variant = _select_variant(variants, location=str(repo_path))
    metadata = Metadata.read_from_file(build_path / variant.variant_str / "metadata.json")

    info = {"path": str(repo_path), "variant": variant.variant_str}
    info.update(_metadata_info(metadata, variants))
    info.update(_dependency_info(metadata, dep_tree=dep_tree))
    return info


def _select_variant(variants: list[Variant], *, location: str) -> Variant:
    variant, trace = resolve_variant(variants)
    if variant is None:
        raise ValueError(f"Cannot find a build variant for this system in {location}:\n\n{variants_trace_str(trace)}")
    return variant


def _dependency_info(metadata: Metadata, *, dep_tree: bool) -> dict:
    # Inspection only reads metadata; it never validates, installs or imports
    # kernel code, so it can also describe untrusted publishers' dependencies.
    resolver = (
        HubCacheResolver(trust_remote_code=True) if constants.HF_HUB_OFFLINE else HubResolver(trust_remote_code=True)
    )
    api = _get_hf_api()
    trees = {
        dep.repo_id: resolve_kernel_tree(api=api, backend=None, kernel=dep, resolver=resolver)
        for dep in metadata.kernel_depends
    }
    kernel_depends: set[str] = set()
    python_depends = set(metadata.python_depends)

    def visit(nodes: dict[str, DepTreeNode]) -> dict:
        tree = {}
        for repo_id, node in sorted(nodes.items()):
            kernel_depends.add(repo_id)
            python_depends.update(node.location.metadata.python_depends)
            tree[repo_id] = visit(node.deps)
        return tree

    tree = visit(trees)
    info: dict[str, Any] = {"kernel_depends": sorted(kernel_depends), "python_depends": sorted(python_depends)}
    if dep_tree:
        info["dependency_tree"] = tree
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
