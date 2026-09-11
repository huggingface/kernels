import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, final

from cryptography.x509 import Certificate
from sigstore.errors import VerificationError
from sigstore.models import Bundle, InvalidBundle
from sigstore.verify import Verifier, policy
from sigstore.verify.policy import VerificationPolicy

from kernels._rust import (
    Digest,
    DigestValidationError,
    DigestViolation,
    KernelLocation,
    Metadata,
    ReceiptError,
    ReceiptStore,
    VerificationReceipt,
)

logger = logging.getLogger(__name__)


class GitHubWorkflowPolicy(VerificationPolicy):
    """
    Accept the signature if it is valid and created from a workflow in the
    given GitHub repository.

    Args:
        signer_uris (`list[str]`):
            List of workflows URIs that are allowed as kernel signers. Each
            URI has the shape:
            `https://github.com/<org>/<repo>/.github/workflows/<workflow>.yaml@<ref>`
    """

    def __init__(
        self,
        *,
        repo_id: str,
        signer_uris: list[str],
    ):
        if not signer_uris:
            raise ValueError("At least one signer URI must be provided")

        self.repo_id = repo_id
        self.signer_uris = signer_uris

    def verify(self, cert: Certificate) -> None:
        policies: list[VerificationPolicy] = [
            policy.OIDCIssuer("https://token.actions.githubusercontent.com"),
            policy.OIDCIssuerV2("https://token.actions.githubusercontent.com"),
            policy.OIDCSourceRepositoryURI(f"https://github.com/{self.repo_id}"),
        ]

        policies.append(policy.AnyOf([policy.OIDCBuildSignerURI(signer_uri) for signer_uri in self.signer_uris]))

        return policy.AllOf(policies).verify(cert)


DEFAULT_POLICY: VerificationPolicy = policy.AnyOf(
    [
        GitHubWorkflowPolicy(
            repo_id="huggingface/kernels-community",
            signer_uris=[
                "https://github.com/huggingface/kernels-community/.github/workflows/build.yaml@refs/heads/main",
                "https://github.com/huggingface/kernels-community/.github/workflows/build-mac.yaml@refs/heads/main",
                "https://github.com/huggingface/kernels-community/.github/workflows/build-windows.yaml@refs/heads/main",
                # This workflow was used to sign existing builds, around Torch 2.10-2.12. Can be removed once these
                # Torch versions are ancient.
                "https://github.com/huggingface/kernels-community/.github/workflows/sign-old-builds.yaml@refs/heads/main",
            ],
        ),
    ]
)
"""
Default verification policy for the kernels package.

Accepts kernels signed by a curated set of trusted kernel developers.
"""


class VerificationResult:
    @final
    @dataclass
    class DigestVerificationFailure:
        """
        Verification failed because there were digest violations.

        The violations are provided through the `violations` field.
        """

        violations: list[DigestViolation]

    @final
    @dataclass
    class MetadataInvalid:
        """
        The kernel metadata could not be parsed.
        """

        reason: str

    @final
    @dataclass
    class SignatureBundleInvalid:
        """
        The signature bundle could not be parsed.
        """

        reason: str

    @final
    @dataclass
    class SignatureVerificationFailure:
        """
        Verification failed because the signature was not valid.
        """

        reason: str

    @final
    @dataclass
    class DigestMissing:
        """
        Verification failed because the metadata did not have a digest.
        """

        pass

    @final
    @dataclass
    class MetadataMissing:
        """
        Verification failed because the kernel did not have metadata.
        """

        pass

    @final
    @dataclass
    class SignatureBundleMissing:
        """
        Verification failed because the kernel metadata was not signed.
        """

        pass

    @final
    @dataclass
    class Success:
        """
        Verification was successful.
        """

        pass

    Any: TypeAlias = (
        DigestMissing
        | DigestVerificationFailure
        | MetadataInvalid
        | MetadataMissing
        | SignatureBundleInvalid
        | SignatureBundleMissing
        | SignatureVerificationFailure
        | Success
    )


def _open_receipt_store() -> ReceiptStore | None:
    """The receipt store, or `None` when verifications cannot be cached."""
    try:
        return ReceiptStore.in_kernels_cache()
    except ReceiptError as e:
        logger.warning(f"Cannot cache kernel verifications: {e}")
        return None


def _has_receipt(store: ReceiptStore, location: KernelLocation) -> bool:
    """Whether the kernel at `location` was verified before.

    An unusable receipt counts as a cache miss: the kernel is then verified in
    full, which overwrites the receipt. A broken cache must never make a kernel
    fail to verify.
    """
    try:
        return store.load(location) is not None
    except ReceiptError as e:
        logger.warning(f"Ignoring unusable kernel verification receipt: {e}")
        return False


def verify_variant(
    variant_path: Path,
    *,
    location: KernelLocation,
    policy: VerificationPolicy | None = None,
    cache: bool = True,
) -> VerificationResult.Any:
    """
    Verify a kernel variant.

    The kernel variant at the given path is verified using a policy. This
    validates that the metadata was signed using a key that is compliant with
    the given policy and that the kernel hashes match the digest in the kernel
    metadata.

    Args:
        variant_path (`Path`):
            Kernel variant path.
        location (`KernelLocation`):
            Identity of the kernel, used to cache the verification.
        policy (`VerificationPolicy`, *optional*):
            Verification policy that should be used while verifying the
            kernel. A default policy that accepts kernels signed by a curated
            set of trusted kernel developers is used if this argument is set
            to `None`.
        cache (`bool`):
            Whether to use the receipt cache to lookup or store kernel
            verifications. Disabling cache use can be useful to validate
            the integrity of a kernel downloaded from the hub.
    """
    verify_policy = DEFAULT_POLICY if policy is None else policy

    bundle_path = variant_path / "metadata.json.sigstore"
    if not bundle_path.is_file():
        return VerificationResult.SignatureBundleMissing()

    try:
        signature_bundle = Bundle.from_json(bundle_path.read_bytes())
    except InvalidBundle as e:
        return VerificationResult.SignatureBundleInvalid(reason=str(e))

    metadata_path = variant_path / "metadata.json"

    if not metadata_path.is_file():
        return VerificationResult.MetadataMissing()

    receipt_store = _open_receipt_store() if cache else None

    if receipt_store is not None and _has_receipt(receipt_store, location):
        # The receipt attests that this kernel metadata was verified
        # using the signature and the kernel data during the digest
        # in the metadata. However, it may have been verified with a
        # different policy, so we have to check certificate in the
        # bundle against the currently required policy.
        try:
            verify_policy.verify(signature_bundle.signing_certificate)
        except VerificationError as e:
            return VerificationResult.SignatureVerificationFailure(reason=str(e))

        return VerificationResult.Success()

    verifier = Verifier.production()

    metadata_bytes = metadata_path.read_bytes()

    # WARNING: always verify the metadata before reading it, this avoids
    #          that a malicious modification can attack the JSON parser.
    try:
        verifier.verify_artifact(
            metadata_bytes,
            signature_bundle,
            verify_policy,
        )
    except VerificationError as e:
        return VerificationResult.SignatureVerificationFailure(reason=str(e))

    try:
        metadata = Metadata.from_bytes(metadata_bytes)
    except (OSError, ValueError) as e:
        return VerificationResult.MetadataInvalid(reason=str(e))

    if metadata.digest is None:
        return VerificationResult.DigestMissing()

    hash = metadata.digest.algorithm

    # Rehash and check that the hashes match up. The validation is delegated to
    # the (Rust) `Digest.validate`, which raises with each individual violation.
    current_digest = Digest.hash_variant(hash, variant_path)
    try:
        metadata.digest.validate(current_digest)
    except DigestValidationError as e:
        return VerificationResult.DigestVerificationFailure(violations=e.violations)

    if receipt_store is not None:
        try:
            receipt_store.store(VerificationReceipt(location))
        except ReceiptError as e:
            logger.warning(f"Cannot store kernel verification receipt: {e}")

    return VerificationResult.Success()
