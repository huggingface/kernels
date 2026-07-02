import json
import sys
from pathlib import Path
from typing import Any

from kernels_data import Metadata

from kernels._versions import _get_available_versions, resolve_version_spec_as_ref
from kernels.utils import CACHE_DIR, _get_hf_api
from kernels.variants import (
    ArchVariant,
    Decision,
    VariantAccepted,
    get_variants,
    get_variants_local,
    resolve_variants,
    variants_trace_str,
)


def print_kernel_info(
    kernel: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    json_output: bool = False,
):
    """Describe a kernel from a local path or a Hub repo ID."""
    path = Path(kernel)
    if path.is_dir():
        if revision is not None or version is not None:
            print("--revision and --version cannot be used with a local path", file=sys.stderr)
            sys.exit(1)
        info, trace = _local_kernel_info(path)
    else:
        info, trace = _hub_kernel_info(kernel, revision=revision, version=version)

    if json_output:
        print(json.dumps(info, indent=2))
    else:
        _print_human(info, trace)


def _hub_kernel_info(
    repo_id: str,
    *,
    revision: str | None,
    version: int | None,
) -> tuple[dict, list[Decision]]:
    if revision is not None and version is not None:
        print("Only one of --revision or --version can be specified", file=sys.stderr)
        sys.exit(1)

    if version is not None:
        revision = resolve_version_spec_as_ref(repo_id, version).name
    elif revision is None:
        versions = _get_available_versions(repo_id)
        revision = versions[max(versions.keys())].name if versions else "main"

    api = _get_hf_api()
    variants = get_variants(api, repo_id=repo_id, revision=revision)
    if not variants:
        print(f"No build variants found in {repo_id} (revision: {revision})", file=sys.stderr)
        sys.exit(1)

    _, trace = resolve_variants(variants, None)

    # The metadata fields that describe the kernel (rather than a specific
    # build) are the same for every variant, so reading one suffices.
    metadata = None
    try:
        metadata_path = api.hf_hub_download(
            repo_id,
            repo_type="kernel",
            filename=f"build/{trace[0].variant.variant_str}/metadata.json",
            cache_dir=CACHE_DIR,
            revision=revision,
        )
        metadata = Metadata.read_from_file(metadata_path)
    except Exception:
        pass

    info = {"repo_id": repo_id, "revision": revision}
    info.update(_metadata_info(metadata, trace))
    return info, trace


def _local_kernel_info(repo_path: Path) -> tuple[dict, list[Decision]]:
    build_path = repo_path / "build"
    variants = get_variants_local(build_path)
    if not variants:
        print(f"No build variants found in: {build_path}", file=sys.stderr)
        sys.exit(1)

    _, trace = resolve_variants(variants, None)

    metadata = None
    for decision in trace:
        metadata_path = build_path / decision.variant.variant_str / "metadata.json"
        if metadata_path.exists():
            metadata = Metadata.read_from_file(metadata_path)
            break

    info = {"path": str(repo_path)}
    info.update(_metadata_info(metadata, trace))
    return info, trace


def _metadata_info(metadata: Metadata | None, trace: list[Decision]) -> dict:
    backends = set()
    for decision in trace:
        variant = decision.variant
        if isinstance(variant, ArchVariant):
            backends.add(variant.arch.backend.name)
        else:
            backends.add(variant.arch.backend_name)

    info: dict[str, Any] = {
        "name": None,
        "version": None,
        "license": None,
        "upstream": None,
        "source": None,
        "python_depends": [],
        "backends": sorted(backends),
        "variants": [
            {
                "variant": decision.variant.variant_str,
                "compatible": isinstance(decision, VariantAccepted),
                "reason": None if isinstance(decision, VariantAccepted) else decision.reason,
            }
            for decision in trace
        ],
    }

    if metadata is not None:
        info.update(
            {
                "name": str(metadata.name),
                "version": metadata.version,
                "license": metadata.license,
                "upstream": metadata.upstream,
                "source": metadata.source,
                "python_depends": metadata.python_depends,
            }
        )

    return info


def _print_human(info: dict, trace: list[Decision]):
    def value(v) -> str:
        return "-" if v is None else str(v)

    if "repo_id" in info:
        print(f"Repository: {info['repo_id']}")
        print(f"Revision: {info['revision']}")
    else:
        print(f"Path: {info['path']}")

    print(f"Name: {value(info['name'])}")
    print(f"Version: {value(info['version'])}")
    print(f"License: {value(info['license'])}")
    print(f"Upstream: {value(info['upstream'])}")
    print(f"Source: {value(info['source'])}")
    print(f"Python dependencies: {', '.join(info['python_depends']) if info['python_depends'] else '-'}")
    print(f"Backends: {', '.join(info['backends'])}")
    print(f"Variants:\n\n{variants_trace_str(trace)}")
