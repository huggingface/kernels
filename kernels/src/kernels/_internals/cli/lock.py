from kernels._internals.compat import tomllib
from kernels._internals.hf_hub import _get_hf_api
from kernels._internals.locking import extract_dependency_locks
from kernels._rust import KernelDependency, KernelVersion


def lock_kernels(args):
    project_dir = args.project_dir

    with open(project_dir / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    kernel_versions = data.get("tool", {}).get("kernels", {}).get("dependencies", None)

    depends = [
        KernelDependency(repo_id=kernel, version=KernelVersion.Version(version))
        for kernel, version in kernel_versions.items()
    ]

    locks = extract_dependency_locks(depends, api=_get_hf_api())

    with open(project_dir / "kernels.lock", "w") as f:
        f.write(locks.to_json())
