"""Type stubs for kernels_data module."""

import os
from enum import Enum
from typing import Optional, final

__all__ = [
    "Backend",
    "BackendInfo",
    "DigestAlgorithm",
    "KernelName",
    "Metadata",
    "Digest",
    "DigestViolation",
    "DigestValidationError",
    "Version",
    "__version__",
]

__version__: str

@final
class Backend(Enum):
    """Kernel backend (hardware target)."""

    CANN = "CANN"
    CPU = "CPU"
    CUDA = "CUDA"
    Metal = "Metal"
    Neuron = "Neuron"
    ROCm = "ROCm"
    XPU = "XPU"

    @staticmethod
    def from_str(s: str) -> "Backend":
        """Parse a backend name.

        Args:
            s: One of ``"cann"``, ``"cpu"``, ``"cuda"``, ``"metal"``,
               ``"neuron"``, ``"rocm"``, ``"xpu"``.

        Raises:
            ValueError: If the backend name is unknown.
        """
        ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class BackendInfo:
    """Backend information."""

    @property
    def backend_type(self) -> Backend:
        """Return the backend type."""
        ...

    @property
    def archs(self) -> Optional[list[str]]:
        """Optional list of target architectures."""
        ...

    def __repr__(self) -> str: ...

@final
class Version:
    """A dotted numeric version (e.g. ``12.8.0``).

    Trailing zeros are stripped during normalization.
    """

    @staticmethod
    def from_str(s: str) -> "Version":
        """Parse a version string of the form ``X``, ``X.Y``, ``X.Y.Z``, ...

        Raises:
            ValueError: If the string is empty or contains non-numeric parts.
        """
        ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, value: object, /) -> bool: ...
    def __lt__(self, value: "Version", /) -> bool: ...
    def __le__(self, value: "Version", /) -> bool: ...
    def __gt__(self, value: "Version", /) -> bool: ...
    def __ge__(self, value: "Version", /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class KernelName:
    """A validated kernel name matching ``^[a-z][-a-z0-9]*[a-z0-9]$``."""

    def __new__(cls, name: str) -> "KernelName":
        """Create a new ``KernelName``.

        Raises:
            ValueError: If the name does not match the required pattern.
        """
        ...

    @property
    def python_name(self) -> str:
        """The name with dashes replaced by underscores."""
        ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class DigestAlgorithm(Enum):
    """Digest algorithm."""

    SHA256 = "SHA256"
    SHA512 = "SHA512"

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class Digest:
    """Source digest for a kernel build variant."""

    @staticmethod
    def hash_variant(
        algorithm: DigestAlgorithm, variant_path: os.PathLike[str] | str
    ) -> "Digest":
        """Hash the files in ``variant_path`` using ``algorithm``.

        Args:
            algorithm: Digest algorithm to use.
            variant_path: Path to the variant directory to hash.

        Raises:
            OSError: If a file cannot be read or the directory cannot be walked.
            RuntimeError: For other unexpected failures.
        """
        ...

    @property
    def algorithm(self) -> DigestAlgorithm:
        """Digest algorithm used."""
        ...

    @property
    def files(self) -> dict[str, str]:
        """Mapping of relative file path to base64-encoded digest."""
        ...

    def validate(self, other: "Digest") -> None:
        """Validate ``other`` (actual) against this digest (expected).

        Returns when the digests match. Otherwise, a `DigestValidationError` is
        raised.

        Raises:
            DigestValidationError: If ``other`` deviates from this digest.
        """
        ...

    def __repr__(self) -> str: ...

class DigestViolation:
    """A violation of a digest when validated against a reference digest.

    This tagged union covers the types of violations. Each violation can be
    converted to a string using ``str(violation)``.
    """

    @final
    class MissingFile(DigestViolation):
        """A file in the reference digest is missing from the digest."""

        path: str
        __match_args__ = ("path",)
        def __new__(cls, path: str) -> "DigestViolation.MissingFile": ...

    @final
    class UnknownFile(DigestViolation):
        """A file present in the digest is not part of the reference digest."""

        path: str
        __match_args__ = ("path",)
        def __new__(cls, path: str) -> "DigestViolation.UnknownFile": ...

    @final
    class HashMismatch(DigestViolation):
        """The hashes for the file differ."""

        path: str
        expected: str
        got: str
        __match_args__ = ("path", "expected", "got")
        def __new__(
            cls, path: str, expected: str, got: str
        ) -> "DigestViolation.HashMismatch": ...

    @final
    class AlgorithmMismatch(DigestViolation):
        """The digest algorithms differ.

        The digest with algorithm ``got`` cannot be validated against the
        reference digest with algorithm ``expected``.
        """

        expected: DigestAlgorithm
        got: DigestAlgorithm
        __match_args__ = ("expected", "got")
        def __new__(
            cls, expected: DigestAlgorithm, got: DigestAlgorithm
        ) -> "DigestViolation.AlgorithmMismatch": ...

    def __str__(self) -> str: ...

class DigestValidationError(Exception):
    """Raised by :meth:`Digest.validate` when a digest cannot be validated against the reference."""

    @property
    def violations(self) -> list[DigestViolation]:
        """The individual digest violations."""
        ...

@final
class Metadata:
    """Parsed ``metadata.json`` for a kernel build variant."""

    @staticmethod
    def read_from_file(metadata_path: os.PathLike[str] | str) -> "Metadata":
        """Parse ``metadata.json`` at the given path.

        Raises:
            ValueError: On any I/O or parse error.
        """
        ...

    @staticmethod
    def from_bytes(bytes: bytes) -> "Metadata":
        """Parse ``metadata.json`` from JSON in a byte array.

        Raises:
            ValueError: On any parse error.
        """
        ...

    @property
    def id(self) -> str: ...
    @property
    def name(self) -> KernelName: ...
    @property
    def version(self) -> Optional[int]: ...
    @property
    def license(self) -> Optional[str]: ...
    @property
    def upstream(self) -> Optional[str]: ...
    @property
    def source(self) -> Optional[str]: ...
    @property
    def python_depends(self) -> list[str]: ...
    @property
    def backend(self) -> BackendInfo: ...
    @property
    def digest(self) -> Optional[Digest]: ...
    def __repr__(self) -> str: ...
