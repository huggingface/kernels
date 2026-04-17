"""Type stubs for kernels_data module."""

from enum import Enum
from typing import Optional, final
import os

__all__ = ["Backend", "KernelName", "Metadata", "Version", "__version__"]

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
class Metadata:
    """Parsed ``metadata.json`` for a kernel build variant."""

    @staticmethod
    def load(metadata_path: os.PathLike[str] | str) -> "Metadata":
        """Parse ``metadata.json`` at the given path.

        Raises:
            ValueError: On any I/O or parse error.
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
