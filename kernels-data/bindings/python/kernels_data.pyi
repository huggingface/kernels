"""Type stubs for kernels_data module."""

import os
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Optional, final

__all__ = [
    "Backend",
    "BackendInfo",
    "Build",
    "General",
    "Provenance",
    "DigestAlgorithm",
    "GitStatus",
    "KernelBuilderVersion",
    "KernelDependency",
    "KernelLock",
    "KernelPaths",
    "KernelLocks",
    "KernelName",
    "KernelVersion",
    "Metadata",
    "NixKernelLock",
    "NixKernelLocks",
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
    TPU = "TPU"
    XPU = "XPU"

    @staticmethod
    def from_str(s: str) -> "Backend":
        """Parse a backend name.

        Args:
            s: One of `"cann"`, `"cpu"`, `"cuda"`, `"metal"`,
               `"neuron"`, `"rocm"`, `"tpu"`, `"xpu"`.

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
class GitStatus:
    """The state of a git working tree."""

    @property
    def commit(self) -> str:
        """Identifier of the `HEAD` commit, as lowercase hexadecimal digits."""
        ...

    @property
    def dirty(self) -> bool:
        """Whether the working tree had uncommitted changes to tracked files."""
        ...

    def __repr__(self) -> str: ...

@final
class KernelBuilderVersion:
    """Provenance of the `kernel-builder` that produced a build."""

    @property
    def version(self) -> str:
        """`kernel-builder` package version."""
        ...

    @property
    def git(self) -> Optional[GitStatus]:
        """Git state of the `kernel-builder` source, when known."""
        ...

    def __repr__(self) -> str: ...

@final
class Provenance:
    """Build provenance: git state of the `kernel-builder` and kernel source."""

    @property
    def kernel_builder(self) -> KernelBuilderVersion:
        """The `kernel-builder` that produced the build (always known)."""
        ...

    @property
    def kernel(self) -> Optional[GitStatus]:
        """Git provenance of the kernel source that was built."""
        ...

    @property
    def dirty(self) -> bool:
        """Whether either the `kernel-builder` or the kernel source was dirty."""
        ...

    def __repr__(self) -> str: ...

@final
class Version:
    """A dotted numeric version (e.g. `12.8.0`).

    Trailing zeros are stripped during normalization.
    """

    @staticmethod
    def from_str(s: str) -> "Version":
        """Parse a version string of the form `X`, `X.Y`, `X.Y.Z`, ...

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
    """A validated kernel name matching `^[a-z][-a-z0-9]*[a-z0-9]$`."""

    def __new__(cls, name: str) -> "KernelName":
        """Create a new `KernelName`.

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
        """Hash the files in `variant_path` using `algorithm`.

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
        """Validate `other` (actual) against this digest (expected).

        Returns when the digests match. Otherwise, a `DigestValidationError` is
        raised.

        Raises:
            DigestValidationError: If `other` deviates from this digest.
        """
        ...

    def __repr__(self) -> str: ...

class DigestViolation:
    """A violation of a digest when validated against a reference digest.

    This tagged union covers the types of violations. Each violation can be
    converted to a string using `str(violation)`.
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

        The digest with algorithm `got` cannot be validated against the
        reference digest with algorithm `expected`.
        """

        expected: DigestAlgorithm
        got: DigestAlgorithm
        __match_args__ = ("expected", "got")
        def __new__(
            cls, expected: DigestAlgorithm, got: DigestAlgorithm
        ) -> "DigestViolation.AlgorithmMismatch": ...

    def __str__(self) -> str: ...

class DigestValidationError(Exception):
    """Raised by `Digest.validate` when a digest cannot be validated against the reference."""

    @property
    def violations(self) -> list[DigestViolation]:
        """The individual digest violations."""
        ...

class KernelVersion:
    """A kernel version: either a numeric version or a git revision string."""

    @final
    class Version(KernelVersion):
        """A numeric kernel version."""

        version: int
        __match_args__ = ("version",)
        def __new__(cls, version: int) -> "KernelVersion.Version": ...

    @final
    class Revision(KernelVersion):
        """A git revision (e.g. commit SHA or branch/tag name)."""

        revision: str
        __match_args__ = ("revision",)
        def __new__(cls, revision: str) -> "KernelVersion.Revision": ...

    def __repr__(self) -> str: ...
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class KernelDependency:
    """A dependency on another kernel."""

    def __new__(cls, repo_id: str, version: KernelVersion) -> "KernelDependency": ...
    @property
    def repo_id(self) -> str:
        """Identifier of the kernel repository this dependency points to."""
        ...

    @property
    def version(self) -> KernelVersion:
        """Version specifier for the dependency."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class KernelLock:
    """A locked kernel revision."""

    def __new__(cls, commit: str) -> "KernelLock":
        """Construct a lock from a commit.

        Raises:
            ValueError: If `commit` is not a full git object id.
        """
        ...

    @property
    def commit(self) -> str:
        """Locked commit of the kernel, as lowercase hexadecimal digits."""
        ...

    @staticmethod
    def from_json(s: str) -> "KernelLock":
        """Parse a `KernelLock` from a JSON string.

        Raises:
            ValueError: If the JSON cannot be parsed.
        """
        ...

    def to_json(self) -> str:
        """Serialize the lock to a pretty-printed JSON string.

        Raises:
            ValueError: If the lock cannot be serialized.
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class KernelLocks:
    """Multiple kernel locks keyed by the dependency they resolve.

    Behaves as a read-only mapping from `KernelDependency` to `KernelLock`.
    """

    def __new__(cls, locks: dict[KernelDependency, KernelLock]) -> "KernelLocks": ...
    def __len__(self) -> int: ...
    def __getitem__(self, dependency: KernelDependency, /) -> KernelLock:
        """Get the lock for `dependency`.

        Raises:
            KeyError: If the dependency is not locked.
        """
        ...

    def __contains__(self, dependency: object, /) -> bool: ...
    def __iter__(self) -> Iterator[KernelDependency]: ...
    def get(
        self, dependency: KernelDependency, default: Optional[KernelLock] = None
    ) -> Optional[KernelLock]:
        """Get the lock for `dependency`, or `default` if it is not locked."""
        ...

    def keys(self) -> list[KernelDependency]:
        """Get the locked dependencies."""
        ...

    def values(self) -> list[KernelLock]:
        """Get the kernel locks."""
        ...

    def items(self) -> list[tuple[KernelDependency, KernelLock]]:
        """Get the (dependency, lock) pairs."""
        ...

    @staticmethod
    def from_json(s: str) -> "KernelLocks":
        """Parse a `KernelLocks` collection from a JSON string.

        Raises:
            ValueError: If the JSON cannot be parsed.
        """
        ...

    def to_json(self) -> str:
        """Serialize the locks collection to a pretty-printed JSON string.

        Raises:
            ValueError: If the locks cannot be serialized.
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class NixKernelLock:
    """A locked kernel revision with the SRI hash of the Nix output path."""

    def __new__(cls, commit: str, hash: str) -> "NixKernelLock":
        """Construct a lock from a commit and a SRI hash.

        Raises:
            ValueError: If `commit` is not a full git object id.
        """
        ...

    @property
    def commit(self) -> str:
        """Locked commit of the kernel, as lowercase hexadecimal digits."""
        ...

    @property
    def hash(self) -> str:
        """SRI hash of the repository snapshot, as used by fixed-output derivations."""
        ...

    @staticmethod
    def from_json(s: str) -> "NixKernelLock":
        """Parse a `NixKernelLock` from a JSON string.

        Raises:
            ValueError: If the JSON cannot be parsed.
        """
        ...

    def to_json(self) -> str:
        """Serialize the lock to a pretty-printed JSON string.

        Raises:
            ValueError: If the lock cannot be serialized.
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class NixKernelLocks:
    """Multiple (Nix) kernel locks keyed by the dependency they resolve.

    This data structure is used to store lock files to be consumed
    by nix-builder.

    Behaves as a read-only mapping from `KernelDependency` to `NixKernelLock`.
    """

    def __new__(
        cls, locks: dict[KernelDependency, NixKernelLock]
    ) -> "NixKernelLocks": ...
    def __len__(self) -> int: ...
    def __getitem__(self, dependency: KernelDependency, /) -> NixKernelLock:
        """Get the lock for `dependency`.

        Raises:
            KeyError: If the dependency is not locked.
        """
        ...

    def __contains__(self, dependency: object, /) -> bool: ...
    def __iter__(self) -> Iterator[KernelDependency]: ...
    def get(
        self, dependency: KernelDependency, default: Optional[NixKernelLock] = None
    ) -> Optional[NixKernelLock]:
        """Get the lock for `dependency`, or `default` if it is not locked."""
        ...

    def keys(self) -> list[KernelDependency]:
        """Get the locked dependencies."""
        ...

    def values(self) -> list[NixKernelLock]:
        """Get the kernel locks."""
        ...

    def items(self) -> list[tuple[KernelDependency, NixKernelLock]]:
        """Get the (dependency, lock) pairs."""
        ...

    @staticmethod
    def from_json(s: str) -> "NixKernelLocks":
        """Parse a `NixKernelLocks` collection from a JSON string.

        Raises:
            ValueError: If the JSON cannot be parsed.
        """
        ...

    def to_json(self) -> str:
        """Serialize the locks collection to a pretty-printed JSON string.

        Raises:
            ValueError: If the locks cannot be serialized.
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class KernelPaths:
    """A collection of kernel paths keyed by the dependency they resolve.

    Behaves as a read-only mapping from `KernelDependency` to `pathlib.Path`.
    """

    def __new__(
        cls, paths: dict[KernelDependency, os.PathLike[str] | str]
    ) -> "KernelPaths": ...
    def __len__(self) -> int: ...
    def __getitem__(self, dependency: KernelDependency, /) -> Path:
        """Get the path for `dependency`.

        Raises:
            KeyError: If the dependency has no path.
        """
        ...

    def __contains__(self, dependency: object, /) -> bool: ...
    def __iter__(self) -> Iterator[KernelDependency]: ...
    def get(
        self,
        dependency: KernelDependency,
        default: Optional[os.PathLike[str] | str] = None,
    ) -> Optional[Path]:
        """Get the path for `dependency`, or `default` if it has no path."""
        ...

    def keys(self) -> list[KernelDependency]:
        """Get the dependencies."""
        ...

    def values(self) -> list[Path]:
        """Get the kernel paths."""
        ...

    def items(self) -> list[tuple[KernelDependency, Path]]:
        """Get the (dependency, path) pairs."""
        ...

    @staticmethod
    def from_json(s: str) -> "KernelPaths":
        """Parse a `KernelPaths` collection from a JSON string.

        Raises:
            ValueError: If the JSON cannot be parsed.
        """
        ...

    def to_json(self) -> str:
        """Serialize the paths collection to a pretty-printed JSON string.

        Raises:
            ValueError: If the paths cannot be serialized.
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class General:
    """General kernel configuration common to all backends."""

    @property
    def backends(self) -> list[Backend]:
        """Backends the kernel supports."""
        ...

    def __repr__(self) -> str: ...

@final
class Build:
    """Parsed and validated `build.toml` configuration for a kernel."""

    @staticmethod
    def open(kernel_dir: os.PathLike[str] | str) -> "Build":
        """Parse and validate the `build.toml` in `kernel_dir`.

        Raises:
            ValueError: If the build configuration cannot be parsed or validated.
        """
        ...

    @property
    def general(self) -> General:
        """General kernel configuration."""
        ...

    def all_kernel_depends(self, backend: Backend) -> list[KernelDependency]:
        """Get the general + backend-specific kernel dependencies for `backend`."""
        ...

@final
class Metadata:
    """Parsed `metadata.json` for a kernel build variant."""

    @staticmethod
    def read_from_file(metadata_path: os.PathLike[str] | str) -> "Metadata":
        """Parse `metadata.json` at the given path.

        Raises:
            ValueError: On any I/O or parse error.
        """
        ...

    @staticmethod
    def from_bytes(bytes: bytes) -> "Metadata":
        """Parse `metadata.json` from JSON in a byte array.

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
    def kernel_depends(self) -> list[KernelDependency]: ...
    @property
    def backend(self) -> BackendInfo: ...
    @property
    def digest(self) -> Optional[Digest]: ...
    @property
    def provenance(self) -> Optional[Provenance]: ...
    def __repr__(self) -> str: ...
