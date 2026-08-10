import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub.dataclasses import strict
from huggingface_hub.hf_api import HfApi
from kernels_data import Build, KernelDependency, Metadata

from kernels._versions import resolve_kernel_version
from kernels.hf_hub import CACHE_DIR, _check_trust_remote_code
from kernels.variants import get_variants


class _JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@strict
@dataclass  # (frozen=True)
class KernelLock:
    repo_id: str
    revision: str
    depends: dict[str, "KernelLock"]


_LOCK_METADATA_CACHE: dict[KernelDependency, KernelLock] = {}


def full_dependency_tree(
    kernel: KernelDependency,
    *,
    api: HfApi,
    backend: str | None,
    seen: set[KernelDependency] | None = None,
) -> KernelLock:

    if (lock := _LOCK_METADATA_CACHE.get(kernel, None)) is not None:
        return lock

    if seen is None:
        seen = set()

    # Check for cycles.
    if kernel in seen:
        raise ValueError(f"Cyclic kernel dependency detected: {kernel.repo_id}")
    seen.add(kernel)

    # Check if the repo is trusted before downloading anything.
    _check_trust_remote_code(
        repo_id=kernel.repo_id,
        local_files_only=False,
        trust_remote_code=False,
    )

    revision = resolve_kernel_version(kernel, local_files_only=False)

    variant_depends = {}

    for variant in get_variants(api, repo_id=kernel.repo_id, revision=revision):
        metadata_path = Path(
            api.hf_hub_download(
                kernel.repo_id,
                repo_type="kernel",
                filename=f"build/{variant.variant_str}/metadata.json",
                cache_dir=CACHE_DIR,
                revision=revision,
                local_files_only=False,
            )
        )
        metadata = Metadata.read_from_file(metadata_path)

        kernel_deps = {}
        for dep in metadata.kernel_depends:
            kernels = kernel_deps[dep.repo_id] = full_dependency_tree(dep, api=api, backend=backend, seen=seen)
            print(kernels)
        variant_depends[variant.variant_str] = kernel_deps

    seen.remove(kernel)

    lock = KernelLock(repo_id=kernel.repo_id, revision=revision, depends=variant_depends)

    _LOCK_METADATA_CACHE[kernel] = lock

    return lock


def print_nix_deps(project_dir: Path):
    build_toml = project_dir / "build.toml"
    if not build_toml.exists():
        raise FileNotFoundError(f"build.toml not found in {project_dir}")

    build = Build.open(project_dir)
    api = HfApi()

    backend_locks = {}

    for backend in build.general.backends:
        kernel_locks = {}
        for kernel in build.all_kernel_depends(backend):
            kernel_locks[kernel.repo_id] = full_dependency_tree(kernel, api=api, backend=str(backend))

        backend_locks[str(backend)] = kernel_locks

    print(json.dumps(backend_locks, cls=_JSONEncoder, indent=2, sort_keys=True))
