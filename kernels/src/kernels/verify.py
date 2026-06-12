from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, final

from cryptography.x509 import Certificate
from kernels_data import Digest, DigestValidationError, DigestViolation, Metadata
from sigstore.errors import VerificationError
from sigstore.models import Bundle, InvalidBundle
from sigstore.verify import Verifier, policy
from sigstore.verify.policy import VerificationPolicy


class GitHubWorkflowPolicy(VerificationPolicy):
    """
    Accept the signature if it is valid and created from a workflow in the
    given GitHub repository.

    Args:
        repo_id (`str`):
            The GitHub repository in the format `owner/repo` that matches
            the workflow that signed the kernel.
    """

    def __init__(self, *, repo_id: str):
        self.repo_id = repo_id

    def verify(self, cert: Certificate) -> None:
        return policy.AllOf(
            [
                policy.OIDCIssuer("https://token.actions.githubusercontent.com"),
                policy.GitHubWorkflowRepository(self.repo_id),
                policy.OIDCIssuerV2("https://token.actions.githubusercontent.com"),
                policy.OIDCSourceRepositoryURI(f"https://github.com/{self.repo_id}"),
            ]
        ).verify(cert)


"""
Default policies for the kernels package.

This is a curated set of trusted kernel developers.
"""
DEFAULT_POLICIES: list[VerificationPolicy] = [
    GitHubWorkflowPolicy(repo_id="huggingface/kernels-community"),
    GitHubWorkflowPolicy(repo_id="huggingface/kernels-staging"),
    GitHubWorkflowPolicy(repo_id="huggingface/kernels-test"),
]


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


def verify_variant(variant_path: Path, policies: list[VerificationPolicy] | None = None) -> VerificationResult.Any:
    """
    Verify a kernel variant.

    The kernel variant at the given path is verified using a set of policies.
    This validates that the metadata was signed using a key that is compliant
    with the given policies and that the kernel hashes match the digest in the
    kernel metadata.

    Args:
        variant_path (`Path`):
            Kernel variant path.
        policies (`list[VerificationPolicy]`, *optional*):
            List of verification policies that should be used while verifying
            the kernel. A default set of policies that accepts kernels signed
            by a curated set of trusted kernel developers is used if this
            argument is set to `None`.
    """
    if policies is None:
        policies = DEFAULT_POLICIES

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

    verifier = Verifier.production()
    verify_policy = policy.AnyOf(policies)

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

    return VerificationResult.Success()
