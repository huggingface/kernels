import sys

from kernels_data import KernelLocks

from kernels.hf_hub import CACHE_DIR, _get_hf_api
from kernels.resolver import _BYTECODE_IGNORE_PATTERNS, resolve_hub_kernel


def download_kernels(args):
    lock_path = args.project_dir / "kernels.lock"

    if not lock_path.exists():
        print(f"No kernels.lock file found in: {args.project_dir}", file=sys.stderr)
        sys.exit(1)

    with open(args.project_dir / "kernels.lock", "r") as f:
        kernel_locks = KernelLocks.from_json(f.read())

    all_successful = True

    api = _get_hf_api()
    for dep, lock in kernel_locks.items():
        print(
            f"Downloading kernel: {dep.repo_id} (revision: {lock.commit})",
            file=sys.stderr,
        )

        try:
            if args.all_variants:
                api.snapshot_download(
                    dep.repo_id,
                    repo_type="kernel",
                    allow_patterns="build/*",
                    ignore_patterns=_BYTECODE_IGNORE_PATTERNS,
                    cache_dir=CACHE_DIR,
                    revision=lock.commit,
                )
            else:
                location = resolve_hub_kernel(dep.repo_id, api=api, backend=None, revision=lock.commit)
                location.install(api=api)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            all_successful = False

    if not all_successful:
        sys.exit(1)
