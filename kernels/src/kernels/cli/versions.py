from kernels._versions import _get_available_versions
from kernels.utils import _get_hf_api
from kernels.variants import (
    get_variants,
    resolve_variants,
)


def print_kernel_versions(repo_id: str):
    api = _get_hf_api()
    versions, repo_type = _get_available_versions(repo_id)
    versions = versions.items()
    if not versions:
        print(f"Repository does not support kernel versions: {repo_id}")
        return

    for version, ref in sorted(versions, key=lambda x: x[0]):
        variants = get_variants(api, repo_id=repo_id, revision=ref.ref, repo_type=repo_type)
        resolved = resolve_variants(variants, None)
        best = resolved[0] if resolved else None
        resolved_set = set(resolved)
        print(f"Version {version}: ", end="")
        variant_strs = [
            (
                f"✅ {variant.variant_str} ({'compatible, preferred' if variant == best else 'compatible'})"
                if variant in resolved_set
                else f"{variant.variant_str}"
            )
            for variant in variants
        ]
        print(", ".join(variant_strs))
