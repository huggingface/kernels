import argparse
import sys

if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing_extensions import assert_never

from kernels._versions import select_revision_or_version
from kernels.utils import install_kernel, install_kernel_all_variants
from kernels.variants import get_variants_local
from kernels.verify import VerificationResult, verify_variant


def verify_signature(args: argparse.Namespace) -> None:
    revision = select_revision_or_version(args.repo_id, revision=None, version=args.version)

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
            case VerificationResult.SignatureBundleMissing():
                if not args.filter_unsigned:
                    print(f"❌ {variant_str}: cannot verify kernel integrity, signature not found")
                    failed = True
                continue
            case VerificationResult.SignatureBundleInvalid(reason=reason):
                if not args.filter_unsigned:
                    print(f"❌ {variant_str}: cannot verify kernel integrity, invalid signature bundle:\n{reason}")
                    failed = True
                continue
            case VerificationResult.MetadataInvalid(reason=reason):
                print(f"❌ {variant_str}: cannot verify kernel integrity, invalid metadata:\n{reason}")
                failed = True
            case VerificationResult.MetadataMissing() | VerificationResult.DigestMissing():
                if not args.filter_no_digest:
                    print(f"❌ {variant_str}: cannot verify kernel integrity, metadata does not have a digest")
                    failed = True
                continue
            case VerificationResult.DigestVerificationFailure(violations=violations):
                print(f"❌ {variant_str}: kernel integrity check failed")
                for violation in violations:
                    print(violation)
                failed = True
            case VerificationResult.SignatureVerificationFailure(reason=reason):
                print(f"❌ {variant_str}: metadata signature verification failed:\n{reason}")
                failed = True
            case VerificationResult.Success():
                print(f"✅ {variant_str}: kernel metadata is correctly signed")
            case _ as unreachable:
                assert_never(unreachable)

    if failed:
        sys.exit(1)
