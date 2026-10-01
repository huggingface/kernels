import sys
from typing import Literal

from huggingface_hub import constants

from kernels._versions import _get_available_versions
from kernels.hf_hub import _get_hf_api
from kernels.variants import (
    VariantAccepted,
    get_variants,
    resolve_variants,
    variants_trace_str,
)


def print_kernel_variants(
    repo_id: str,
    *,
    all_versions: bool = False,
    only_compatible: bool = False,
    version: int | Literal["latest"] | None = None,
    revision: str | None = None,
):
    """Print build variants and compatibility decisions for selected versions."""
    if sum((all_versions, version is not None, revision is not None)) > 1:
        print("Only one of --all-versions, --version, or --revision can be specified", file=sys.stderr)
        sys.exit(1)

    if revision is not None:
        revisions = [(f"Revision {revision}", revision)]
    else:
        versions = _get_available_versions(repo_id, local_files_only=constants.HF_HUB_OFFLINE)
        if version is not None and version != "latest":
            if version not in versions:
                print(
                    f"Version {version} not found, available versions: {', '.join(str(v) for v in sorted(versions))}",
                    file=sys.stderr,
                )
                sys.exit(1)
            selected_versions = [version]
        elif not versions:
            print(f"Repository does not support kernel versions: {repo_id}", file=sys.stderr)
            sys.exit(1)
        else:
            selected_versions = sorted(versions) if all_versions else [max(versions)]
        revisions = [(f"Version {v}", versions[v].ref) for v in selected_versions]

    api = _get_hf_api()
    for label, ref in revisions:
        variants = get_variants(api, repo_id=repo_id, revision=ref)
        _, status = resolve_variants(variants, None)
        if only_compatible:
            status = [decision for decision in status if isinstance(decision, VariantAccepted)]
        output = variants_trace_str(status)
        if not output:
            output = "No compatible variants found." if only_compatible else "No build variants found."
        print(f"{label}:\n\n{output}")
