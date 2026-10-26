import json
import sys
from pathlib import Path
from typing import cast

from huggingface_hub import CachedRevisionInfo, constants, scan_cache_dir
from huggingface_hub.utils import CacheNotFound, tabulate
from huggingface_hub.utils._parsing import format_timesince

from kernels.hf_hub import _get_cache_dir
from kernels.variants import get_variants_local


def print_cached_kernels(
    *,
    cache_dir: Path | None = None,
    json_output: bool = False,
    show_warnings: bool = False,
    no_truncate: bool = False,
) -> None:
    """List cached kernel versions with their own build variants."""
    cache_dir = Path(cache_dir or _get_cache_dir() or constants.HF_HUB_CACHE).expanduser().resolve()
    try:
        cache = scan_cache_dir(cache_dir)
    except CacheNotFound:
        cache = None
    except (OSError, ValueError) as error:
        print(f"Cannot scan kernel cache: {error}", file=sys.stderr)
        sys.exit(1)

    repos = []
    rows: list[list[str]] = []
    if cache is not None:
        for repo in sorted(cache.repos, key=lambda repo: repo.repo_id):
            if repo.repo_type != "kernel":
                continue
            version_refs = [
                (number, ref, revision)
                for ref, revision in repo.refs.items()
                if (number := _version_number(ref)) is not None
            ]
            if not version_refs:
                continue
            versions = []
            for _, version, revision in sorted(version_refs, key=lambda entry: (entry[0], entry[1])):
                variants = _cached_variants(revision)
                last_accessed = max(
                    (file.blob_last_accessed for file in revision.files),
                    default=revision.snapshot_path.stat().st_atime,
                )
                versions.append(
                    {
                        "version": version,
                        "size_on_disk": revision.size_on_disk,
                        "last_accessed": last_accessed,
                        "last_modified": revision.last_modified,
                        "variants": variants,
                    }
                )
                rows.append(
                    [
                        f"kernel/{repo.repo_id}",
                        revision.size_on_disk_str,
                        format_timesince(last_accessed),
                        revision.last_modified_str,
                        version,
                        variants[0] if variants else "-",
                    ]
                )
                rows.extend(["", "", "", "", "", variant] for variant in variants[1:])
            repos.append(
                {
                    "repo_id": repo.repo_id,
                    "size_on_disk": repo.size_on_disk,
                    "last_accessed": repo.last_accessed,
                    "last_modified": repo.last_modified,
                    "versions": versions,
                }
            )

    if json_output:
        print(json.dumps({"cache_dir": str(cache_dir), "repos": repos}, indent=2))
    elif not rows:
        print(f"No cached kernel versions found in {cache_dir}.")
    else:
        # Use the same terminal-width handling as `hf cache ls`.
        from huggingface_hub.cli._output import _truncate_columns

        headers = ["ID", "SIZE", "LAST_ACCESSED", "LAST_MODIFIED", "VERSION", "VARIANTS"]
        truncated = _truncate_columns(headers, rows, no_truncate=no_truncate)
        print(tabulate(cast(list[list[str | int]], rows), headers=headers, alignments={"SIZE": "right"}))
        if truncated:
            print("Hint: Use `--no-truncate` or `--json` to display full values.", file=sys.stderr)

    if cache is not None and cache.warnings:
        if show_warnings:
            for warning in cache.warnings:
                print(f"Warning: {str(warning).rstrip('.')}. Repo ignored.", file=sys.stderr)
        else:
            print(
                f"Warning: Found {len(cache.warnings)} cache inconsistencies. "
                "Re-run with `--show-warnings` to display them.",
                file=sys.stderr,
            )


def _version_number(ref: str) -> int | None:
    if ref.startswith("v") and ref[1:].isdecimal():
        return int(ref[1:])
    return None


def _cached_variants(revision: CachedRevisionInfo) -> list[str]:
    build_path = revision.snapshot_path / "build"
    # Empty build directories do not contain a cached variant.
    cached_dirs = {
        file.file_path.relative_to(build_path).parts[0]
        for file in revision.files
        if build_path in file.file_path.parents
    }
    return sorted(
        variant.variant_str for variant in get_variants_local(build_path) if variant.variant_str in cached_dirs
    )
