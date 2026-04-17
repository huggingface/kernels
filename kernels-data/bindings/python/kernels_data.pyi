"""Type stubs for kernels_data module."""

from enum import Enum
from typing import Optional
import os

__version__: str

class Backend(Enum):
    """Kernel backend (hardware target)."""

    CPU = "CPU"
    CUDA = "CUDA"
    METAL = "METAL"
    NEURON = "NEURON"
    ROCM = "ROCM"
    XPU = "XPU"

    @staticmethod
    def from_str(s: str) -> "Backend":
        """Parse a backend name.

        Args:
            s: One of ``"cpu"``, ``"cuda"``, ``"metal"``, ``"neuron"``,
               ``"rocm"``, ``"xpu"``.

        Raises:
            ValueError: If the backend name is unknown.
        """
        ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

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
    def __eq__(self, other: object) -> bool: ...
    def __lt__(self, other: "Version") -> bool: ...
    def __le__(self, other: "Version") -> bool: ...
    def __gt__(self, other: "Version") -> bool: ...
    def __ge__(self, other: "Version") -> bool: ...
    def __hash__(self) -> int: ...

class KernelName:
    """A validated kernel name matching ``^[a-z][-a-z0-9]*[a-z0-9]$``."""

    def __init__(self, name: str) -> None:
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
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class Metadata:
    """Parsed ``metadata.json`` for a kernel build variant."""

    @staticmethod
    def load_from_variant(variant_path: os.PathLike[str] | str) -> Optional["Metadata"]:
        """Load ``metadata.json`` from a build variant directory.

        Returns ``None`` if the file does not exist in ``variant_path``.

        Raises:
            ValueError: If the file exists but cannot be parsed.
        """
        ...

    @property
    def version(self) -> Optional[int]: ...
    @property
    def license(self) -> Optional[str]: ...
    @property
    def upstream(self) -> Optional[str]: ...
    @property
    def python_depends(self) -> list[str]: ...
    @property
    def backend(self) -> Backend: ...
    def __repr__(self) -> str: ...

def parse_metadata(path: os.PathLike[str] | str) -> Metadata:
    """Parse a kernel ``metadata.json`` file.

    Raises:
        ValueError: On any I/O or parse error.
    """
    ...
