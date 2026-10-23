import json
import sys
from pathlib import Path

from huggingface_hub import CachedRepoInfo, constants, scan_cache_dir
from huggingface_hub.utils import CacheNotFound, tabulate

from kernels.hf_hub import _get_cache_dir
from kernels.variants import get_variants_local


def print_cached_kernels(*, cache_dir: Path | None = None, json_output: bool = False) -> None:
    """List cached kernel repositories and all their cached build variants."""
    cache_dir = Path(cache_dir or _get_cache_dir() or constants.HF_HUB_CACHE).expanduser().resolve()
    try:
        cache = scan_cache_dir(cache_dir)
    except CacheNotFound:
        cache = None
    except (OSError, ValueError) as error:
        print(f"Cannot scan kernel cache: {error}", file=sys.stderr)
        sys.exit(1)

    repos = []
    rows: list[list[str | int]] = []
    if cache is not None:
        for warning in cache.warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        for repo in sorted(cache.repos, key=lambda repo: repo.repo_id):
            if repo.repo_type != "kernel":
                continue
            variants = _cached_variants(repo)
            refs = sorted(repo.refs)
            repos.append(
                {
                    "repo_id": repo.repo_id,
                    "size_on_disk": repo.size_on_disk,
                    "last_accessed": repo.last_accessed,
                    "last_modified": repo.last_modified,
                    "refs": refs,
                    "variants": variants,
                }
            )
            rows.append(
                [
                    f"kernel/{repo.repo_id}",
                    repo.size_on_disk_str,
                    repo.last_accessed_str,
                    repo.last_modified_str,
                    ", ".join(refs) or "-",
                    ", ".join(variants) or "-",
                ]
            )

    if json_output:
        print(json.dumps({"cache_dir": str(cache_dir), "repos": repos}, indent=2))
    elif not rows:
        print(f"No cached kernels found in {cache_dir}.")
    else:
        print(tabulate(rows, headers=["ID", "SIZE", "LAST_ACCESSED", "LAST_MODIFIED", "REFS", "VARIANTS"]))


def _cached_variants(repo: CachedRepoInfo) -> list[str]:
    names: set[str] = set()
    for revision in repo.revisions:
        build_path = revision.snapshot_path / "build"
        # Empty build directories do not contain a cached variant.
        cached_dirs = {
            file.file_path.relative_to(build_path).parts[0]
            for file in revision.files
            if build_path in file.file_path.parents
        }
        names.update(
            variant.variant_str for variant in get_variants_local(build_path) if variant.variant_str in cached_dirs
        )
    return sorted(names)
