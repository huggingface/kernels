from pathlib import Path

from huggingface_hub.hf_api import HfApi
from kernels_data import Build

from kernels.locking import KernelLocks, extract_dependency_locks


def print_lock_kernel_depends(project_dir: Path):
    build_toml = project_dir / "build.toml"
    if not build_toml.exists():
        raise FileNotFoundError(f"build.toml not found in {project_dir}")

    build = Build.open(project_dir)
    api = HfApi()

    kernel_locks = {}

    for backend in build.general.backends:
        for dep, lock in extract_dependency_locks(
            build.all_kernel_depends(backend), api=api, backend=str(backend)
        ).items():
            kernel_locks[dep] = lock

    print(KernelLocks(locks=kernel_locks).to_json())
