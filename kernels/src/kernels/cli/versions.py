from importlib.util import find_spec
from pathlib import Path
from huggingface_hub import HfApi

from kernels._versions import _get_available_versions
from kernels.utils import _get_hf_api, build_variants
from kernels.variants import BUILD_VARIANT_REGEX


def print_kernel_versions(repo_id: str):
    api = _get_hf_api()

    if find_spec("torch") is None:
        # Do not mark compatible variants when Torch is not available.
        compatible_variants = set()
    else:
        compatible_variants = set(build_variants())

    versions = _get_available_versions(repo_id).items()
    if not versions:
        print(f"Repository does not support kernel versions: {repo_id}")
        return

    for version, ref in sorted(versions, key=lambda x: x[0]):
        print(f"Version {version}: ", end="")
        variants = [
            f"{variant} ✅" if variant in compatible_variants else f"{variant}"
            for variant in _get_build_variants(api, repo_id, ref.ref)
        ]
        print(", ".join(variants))


def _get_build_variants(api: "HfApi", repo_id: str, revision: str) -> list[str]:
    variants = set()
    for filename in api.list_repo_files(repo_id, revision=revision):
        path = Path(filename)
        if len(path.parts) < 2 or path.parts[0] != "build":
            continue

        match = BUILD_VARIANT_REGEX.match(path.parts[1])
        if match:
            variants.add(path.parts[1])
    return sorted(variants)
