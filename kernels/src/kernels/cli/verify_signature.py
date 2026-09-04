import argparse
import sys

from huggingface_hub import constants

from kernels._versions import select_revision_or_version
from kernels.install import install_kernel, install_kernel_all_variants
from kernels.variants import get_variants_local
from kernels.verify import VerificationResult, verify_variant


def verify_signature(args: argparse.Namespace) -> None:
    revision = select_revision_or_version(
        args.repo_id,
        revision=None,
        version=args.version,
        local_files_only=constants.HF_HUB_OFFLINE,
    )

    if args.all_variants:
        repo_path = install_kernel_all_variants(args.repo_id, revision=revision)
        variants = get_variants_local(repo_path)
        kernel_paths = [repo_path / variant.variant_str for variant in variants]
    else:
        kernel_paths = [install_kernel(args.repo_id, revision=revision)]

    failed = False

    for kernel_path in kernel_paths:
        result = verify_variant(kernel_path)
        variant_str = kernel_path.name

        match result:
            case VerificationResult.SignatureBundleMissing() if args.filter_unsigned:
                continue
            case VerificationResult.MetadataMissing() | VerificationResult.DigestMissing() if args.filter_no_digest:
                continue
            case VerificationResult.Success():
                print(f"✅ {variant_str}: {result}")
            case _:
                print(f"❌ {variant_str}: {result}")
                failed = True

    if failed:
        sys.exit(1)
