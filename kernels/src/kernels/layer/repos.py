import sys
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, Type, runtime_checkable

from ._interval_tree import IntervalTree
from .device import CUDAProperties, Device, ROCMProperties
from .mode import Mode

if TYPE_CHECKING:
    from torch import nn


class RepositoryProtocol(Protocol):
    def load(self) -> Type["nn.Module"]: ...


class DeviceRepos(ABC):
    """
    Device-specific kernel layer repositories.
    """

    @staticmethod
    def create_repo(device: Device) -> "DeviceRepos":
        """Create an appropriate repository set for this device type."""
        if device.type == "cpu":
            return _CPURepos()
        elif device.type == "cuda":
            return _CUDARepos()
        elif device.type == "rocm":
            return _ROCMRepos()
        elif device.type == "mps":
            return _MPSRepos()
        elif device.type == "xpu":
            return _XPURepos()
        elif device.type == "npu":
            return _NPURepos()
        elif device.type == "neuron":
            return _NeuronRepos()
        elif device.type == "tpu":
            return _TPURepos()
        else:
            raise ValueError(f"Unknown device type: {device.type}")

    @property
    @abstractmethod
    def repos(
        self,
    ) -> dict[Mode, RepositoryProtocol] | None: ...

    @abstractmethod
    def insert(self, device: Device, repos: dict[Mode, RepositoryProtocol]):
        """
        Insert a repository for a specific device and mode.
        """
        ...


class _CPURepos(DeviceRepos):
    _repos: dict[Mode, RepositoryProtocol]

    def __init__(self):
        super().__init__()
        self._repos = {}

    @property
    def repos(
        self,
    ) -> dict[Mode, RepositoryProtocol] | None:
        return self._repos

    def insert(self, device: Device, repos: dict[Mode, RepositoryProtocol]):
        if device.type != "cpu":
            raise ValueError(f"Device type must be 'cpu', got {device.type}")

        self._repos = repos


class _XPURepos(DeviceRepos):
    _repos: dict[Mode, RepositoryProtocol]

    def __init__(self):
        super().__init__()
        self._repos = {}

    @property
    def repos(
        self,
    ) -> dict[Mode, RepositoryProtocol] | None:
        return self._repos

    def insert(self, device: Device, repos: dict[Mode, RepositoryProtocol]):
        if device.type != "xpu":
            raise ValueError(f"Device type must be 'xpu', got {device.type}")

        self._repos = repos


class _NeuronRepos(DeviceRepos):
    _repos: dict[Mode, RepositoryProtocol]

    def __init__(self):
        super().__init__()
        self._repos = {}

    @property
    def repos(
        self,
    ) -> dict[Mode, RepositoryProtocol] | None:
        return self._repos

    def insert(self, device: Device, repos: dict[Mode, RepositoryProtocol]):
        if device.type != "neuron":
            raise ValueError(f"Device type must be 'neuron', got {device.type}")

        self._repos = repos


class _TPURepos(DeviceRepos):
    _repos: dict[Mode, RepositoryProtocol]

    def __init__(self):
        super().__init__()
        self._repos = {}

    @property
    def repos(
        self,
    ) -> dict[Mode, RepositoryProtocol] | None:
        return self._repos

    def insert(self, device: Device, repos: dict[Mode, RepositoryProtocol]):
        if device.type != "tpu":
            raise ValueError(f"Device type must be 'tpu', got {device.type}")

        self._repos = repos


class _NPURepos(DeviceRepos):
    _repos: dict[Mode, RepositoryProtocol]

    def __init__(self):
        super().__init__()
        self._repos = {}

    @property
    def repos(
        self,
    ) -> dict[Mode, RepositoryProtocol] | None:
        return self._repos

    def insert(self, device: Device, repos: dict[Mode, RepositoryProtocol]):
        if device.type != "npu":
            raise ValueError(f"Device type must be 'npu', got {device.type}")

        self._repos = repos


class _MPSRepos(DeviceRepos):
    _repos: dict[Mode, RepositoryProtocol]

    def __init__(self):
        super().__init__()
        self._repos = {}

    @property
    def repos(
        self,
    ) -> dict[Mode, RepositoryProtocol] | None:
        return self._repos

    def insert(self, device: Device, repos: dict[Mode, RepositoryProtocol]):
        if device.type != "mps":
            raise ValueError(f"Device type must be 'mps', got {device.type}")

        self._repos = repos


class _CUDARepos(DeviceRepos):
    _repos: IntervalTree[dict[Mode, RepositoryProtocol]]

    def __init__(self):
        super().__init__()
        self.repos_by_capability = IntervalTree()

    @property
    def repos(
        self,
    ) -> dict[Mode, RepositoryProtocol] | None:
        capability = _find_capability()
        return self.repos_by_capability.find_smallest_interval(capability)

    def insert(self, device: Device, repos: dict[Mode, RepositoryProtocol]):
        assert device.properties is None or isinstance(device.properties, CUDAProperties)

        min_capability = 0 if device.properties is None else device.properties.min_capability
        max_capability = sys.maxsize if device.properties is None else device.properties.max_capability

        self.repos_by_capability.insert(min_capability, max_capability, repos)


class _ROCMRepos(DeviceRepos):
    _repos: IntervalTree[dict[Mode, RepositoryProtocol]]

    def __init__(self):
        super().__init__()
        self.repos_by_capability = IntervalTree()

    @property
    def repos(
        self,
    ) -> dict[Mode, RepositoryProtocol] | None:
        capability = _find_capability()
        return self.repos_by_capability.find_smallest_interval(capability)

    def insert(self, device: Device, repos: dict[Mode, RepositoryProtocol]):
        assert device.properties is None or isinstance(device.properties, ROCMProperties)

        min_capability = 0 if device.properties is None else device.properties.min_capability
        max_capability = sys.maxsize if device.properties is None else device.properties.max_capability

        self.repos_by_capability.insert(min_capability, max_capability, repos)


_MODE_FALLBACK_PRIORITY = {
    Mode.INFERENCE: [
        Mode.INFERENCE,
        Mode.INFERENCE | Mode.TORCH_COMPILE,
        Mode.TRAINING,
        Mode.TRAINING | Mode.TORCH_COMPILE,
        Mode.FALLBACK,
    ],
    Mode.TRAINING: [
        Mode.TRAINING,
        Mode.TRAINING | Mode.TORCH_COMPILE,
        Mode.FALLBACK,
    ],
    Mode.INFERENCE | Mode.TORCH_COMPILE: [
        Mode.INFERENCE | Mode.TORCH_COMPILE,
        Mode.TRAINING | Mode.TORCH_COMPILE,
        Mode.FALLBACK,
    ],
    Mode.TRAINING | Mode.TORCH_COMPILE: [
        Mode.TRAINING | Mode.TORCH_COMPILE,
        Mode.FALLBACK,
    ],
}


def _select_repository(
    repositories: dict[Mode, RepositoryProtocol],
    *,
    mode: Mode,
) -> tuple[RepositoryProtocol, Mode] | None:
    # Get the fallback priority list for the requested mode
    if mode not in _MODE_FALLBACK_PRIORITY:
        raise ValueError(f"Unsupported mode: {mode}")

    fallback_modes = _MODE_FALLBACK_PRIORITY[mode]

    # Try each mode in priority order
    for fallback_mode in fallback_modes:
        if fallback_mode in repositories:
            return (repositories[fallback_mode], fallback_mode)

    return None


@runtime_checkable
class KernelLayerSelectorProtocol(Protocol):
    """
    Callable that selects the kernel repository for a layer at kernelization time.

    A selector can be used in a kernel mapping instead of the per-device dictionary. [`kernelize`] calls
    the selector for every module with the mapped layer name, so the selector can base its decision on
    the module instance itself, the device type, and the kernelization mode.

    Selectors should be stateless and must not hold references to modules; the selection should only depend
    on the `module`, `device_type`, and `mode` arguments.

    An example can be found in the documentation of [`use_kernel_mapping`].
    """

    def __call__(
        self, module: "nn.Module", *, device_type: Device, mode: Mode
    ) -> tuple[RepositoryProtocol, Mode] | None:
        """
        Select the kernel repository for a module.

        Args:
            module (`nn.Module`):
                The module that is being kernelized.
            device_type ([`Device`]):
                The device that kernels are loaded for.
            mode ([`Mode`]):
                The mode that the module is kernelized for.

        Returns:
            `tuple[RepositoryProtocol, Mode] | None`: The repository and the mode that it supports, or `None`
            when no kernel should be used for the module.
        """
        ...


@lru_cache
def _find_capability() -> int:
    import torch

    major, minor = torch.cuda.get_device_capability(device=None)
    return major * 10 + minor
