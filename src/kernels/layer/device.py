from dataclasses import dataclass
from typing import (
    Optional,
)


@dataclass(frozen=True)
class CUDAProperties:
    """
    CUDA-specific device properties for capability-based kernel selection.

    This class defines CUDA compute capability constraints for kernel selection, allowing kernels to specify
    minimum and maximum CUDA compute capabilities they support.

    Args:
        min_capability (`int`):
            Minimum CUDA compute capability required (e.g., 75 for compute capability 7.5).
        max_capability (`int`):
            Maximum CUDA compute capability supported (e.g., 90 for compute capability 9.0).

    Example:
        ```python
        from kernels import CUDAProperties, Device

        # Define CUDA properties for modern GPUs (compute capability 7.5 to 9.0)
        cuda_props = CUDAProperties(min_capability=75, max_capability=90)

        # Create a device with these properties
        device = Device(type="cuda", properties=cuda_props)
        ```

    Note:
        CUDA compute capabilities are represented as integers where the major and minor versions are concatenated.
        For example, compute capability 7.5 is represented as 75, and 8.6 is represented as 86.
    """

    min_capability: int
    max_capability: int

    def __eq__(self, other):
        if not isinstance(other, CUDAProperties):
            return NotImplemented
        return (
            self.min_capability == other.min_capability
            and self.max_capability == other.max_capability
        )

    def __hash__(self):
        return hash((self.min_capability, self.max_capability))


@dataclass(frozen=True)
class ROCMProperties:
    """
    ROCM-specific device properties for capability-based kernel selection.

    This class defines ROCM compute capability constraints for kernel selection, allowing kernels to specify
    minimum and maximum ROCM compute capabilities they support.

    Args:
        min_capability (`int`):
            Minimum ROCM compute capability required (e.g., 75 for compute capability 7.5).
        max_capability (`int`):
            Maximum ROCM compute capability supported (e.g., 90 for compute capability 9.0).

    Example:
        ```python
        from kernels import ROCMProperties, Device

        # Define ROCM properties for modern GPUs (compute capability 7.5 to 9.0)
        rocm_props = ROCMProperties(min_capability=75, max_capability=90)

        # Create a device with these properties
        device = Device(type="rocm", properties=rocm_props)
        ```

    Note:
        ROCM compute capabilities are represented as integers where the major and minor versions are concatenated.
        For example, compute capability 7.5 is represented as 75, and 8.6 is represented as 86.
    """

    min_capability: int
    max_capability: int

    def __eq__(self, other):
        if not isinstance(other, ROCMProperties):
            return NotImplemented
        return (
            self.min_capability == other.min_capability
            and self.max_capability == other.max_capability
        )

    def __hash__(self):
        return hash((self.min_capability, self.max_capability))


@dataclass(frozen=True)
class Device:
    """
    Represents a compute device with optional properties.

    This class encapsulates device information including device type and optional device-specific properties
    like CUDA capabilities.

    Args:
        type (`str`):
            The device type (e.g., "cuda", "mps", "npu", "rocm", "xpu").
        properties ([`CUDAProperties`], *optional*):
            Device-specific properties. Currently only [`CUDAProperties`] is supported for CUDA devices.

    Example:
        ```python
        from kernels import Device, CUDAProperties

        # Basic CUDA device
        cuda_device = Device(type="cuda")

        # CUDA device with specific capability requirements
        cuda_device_with_props = Device(
            type="cuda",
            properties=CUDAProperties(min_capability=75, max_capability=90)
        )

        # MPS device for Apple Silicon
        mps_device = Device(type="mps")

        # XPU device (e.g., Intel(R) Data Center GPU Max 1550)
        xpu_device = Device(type="xpu")

        # NPU device (Huawei Ascend)
        npu_device = Device(type="npu")
        ```
    """

    type: str
    properties: Optional[CUDAProperties] = None

    def __post_init__(self):
        if self.properties is not None and isinstance(self.properties, CUDAProperties):
            if self.type != "cuda":
                raise ValueError("CUDAProperties is only supported for 'cuda' devices.")

    def __eq__(self, other):
        if not isinstance(other, Device):
            return NotImplemented
        return self.type == other.type and self.properties == other.properties

    def __hash__(self):
        return hash((self.type, self.properties))
