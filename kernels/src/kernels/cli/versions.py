from kernels._versions import _get_available_versions
from kernels.utils import _get_hf_api
from kernels.variants import (
    get_variants,
    resolve_variants,
    variants_trace_str,
)


def print_kernel_versions(repo_id: str):
    api = _get_hf_api()
    versions = _get_available_versions(repo_id)
    if not versions:
        print(f"Repository does not support kernel versions: {repo_id}")
        return

    for version, ref in sorted(versions.items(), key=lambda x: x[0]):
        variants = get_variants(api, repo_id=repo_id, revision=ref.ref)
        _, status = resolve_variants(variants, None)
        print(f"Version {version}:\n\n{variants_trace_str(status)}")
