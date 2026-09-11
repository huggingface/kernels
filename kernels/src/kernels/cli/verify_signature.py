import argparse
import sys

from huggingface_hub import constants

if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing_extensions import assert_never

from kernels._rust import KernelLocation
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
        variant_str = kernel_path.name

        result = verify_variant(
            kernel_path,
            location=KernelLocation.remote(args.repo_id, revision, variant_str),
            # Always fully verify the kernel in this subcommand.
            cache=False,
        )

        match result:
            case VerificationResult.SignatureBundleMissing() if args.filter_unsigned:
                pass
            case VerificationResult.MetadataMissing() | VerificationResult.DigestMissing() if args.filter_no_digest:
                pass
            case VerificationResult.Success():
                print(f"✅ {variant_str}: {result}")
            case VerificationResult.Failure():
                print(f"❌ {variant_str}: {result}")
                failed = True
            case _ as unreachable:
                assert_never(unreachable)

    if failed:
        sys.exit(1)
