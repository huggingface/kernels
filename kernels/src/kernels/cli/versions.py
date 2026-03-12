from kernels._versions import _get_available_versions
from kernels.utils import _get_hf_api
from kernels.variants import get_variants, resolve_variant


def print_kernel_versions(repo_id: str):
    api = _get_hf_api()

    versions = _get_available_versions(repo_id).items()
    if not versions:
        print(f"Repository does not support kernel versions: {repo_id}")
        return

    for version, ref in sorted(versions, key=lambda x: x[0]):
        variants = get_variants(api, repo_id=repo_id, revision=ref.ref)
        best = resolve_variant(variants)
        print(f"Version {version}: ", end="")
        variant_strs = [
            f"{variant.variant_str} ✅" if variant == best else f"{variant.variant_str}"
            for variant in variants
        ]
        print(", ".join(variant_strs))
