from huggingface_hub.hf_api import HfApi
from kernels_data import Build, KernelDependency, KernelLocks, KernelVersion

from kernels.compat import tomllib
from kernels.hf_hub import _get_hf_api
from kernels.locking import extract_dependency_locks


def lock_kernels(args):
    project_dir = args.project_dir

    if args.kernel:
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
    else:
        with open(project_dir / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)

        kernel_versions = data.get("tool", {}).get("kernels", {}).get("dependencies", None)

        depends = [
            KernelDependency(repo_id=kernel, version=KernelVersion.Version(version))
            for kernel, version in kernel_versions.items()
        ]

        locks = extract_dependency_locks(depends, api=_get_hf_api(), backend=None)

        with open(project_dir / "kernels.lock", "w") as f:
            f.write(locks.to_json())
