import dataclasses
import json
from pathlib import Path

from huggingface_hub.hf_api import HfApi
from kernels_data import Build

from kernels.locking import extract_dependency_locks


class _JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def print_lock_kernel_depends(project_dir: Path):
    build_toml = project_dir / "build.toml"
    if not build_toml.exists():
        raise FileNotFoundError(f"build.toml not found in {project_dir}")

    build = Build.open(project_dir)
    api = HfApi()

    backend_locks = {}

    for backend in build.general.backends:
        backend_locks[str(backend)] = extract_dependency_locks(
            build.all_kernel_depends(backend), api=api, backend=str(backend)
        )

    print(json.dumps(backend_locks, cls=_JSONEncoder, indent=2, sort_keys=True))
