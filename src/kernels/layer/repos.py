from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Protocol, Tuple, Type
import sys
from functools import lru_cache

from .device import Device
from .mode import Mode
from ._interval_tree import IntervalTree
from .device import CUDAProperties, ROCMProperties

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
        else:
            raise ValueError(f"Unknown device type: {device.type}")

    @property
    @abstractmethod
    def repos(
        self,
    ) -> Optional[Dict[Mode, RepositoryProtocol]]: ...

    @abstractmethod
    def insert(self, device: Device, repos: Dict[Mode, RepositoryProtocol]):
        """
        Insert a repository for a specific device and mode.
        """
        ...


class _CPURepos(DeviceRepos):
    _repos: Dict[Mode, RepositoryProtocol]

    def __init__(self):
        super().__init__()
        self._repos = {}

    @property
    def repos(
        self,
    ) -> Optional[Dict[Mode, RepositoryProtocol]]:
        return self._repos

    def insert(self, device: Device, repos: Dict[Mode, RepositoryProtocol]):
        if device.type != "cpu":
            raise ValueError(f"Device type must be 'cpu', got {device.type}")

        self._repos = repos


class _XPURepos(DeviceRepos):
    _repos: Dict[Mode, RepositoryProtocol]

    def __init__(self):
        super().__init__()
        self._repos = {}

    @property
    def repos(
        self,
    ) -> Optional[Dict[Mode, RepositoryProtocol]]:
        return self._repos

    def insert(self, device: Device, repos: Dict[Mode, RepositoryProtocol]):
        if device.type != "xpu":
            raise ValueError(f"Device type must be 'xpu', got {device.type}")

        self._repos = repos


class _NPURepos(DeviceRepos):
    _repos: Dict[Mode, RepositoryProtocol]

    def __init__(self):
        super().__init__()
        self._repos = {}

    @property
    def repos(
        self,
    ) -> Optional[Dict[Mode, RepositoryProtocol]]:
        return self._repos

    def insert(self, device: Device, repos: Dict[Mode, RepositoryProtocol]):
        if device.type != "npu":
            raise ValueError(f"Device type must be 'npu', got {device.type}")

        self._repos = repos


class _MPSRepos(DeviceRepos):
    _repos: Dict[Mode, RepositoryProtocol]

    def __init__(self):
        super().__init__()
        self._repos = {}

    @property
    def repos(
        self,
    ) -> Optional[Dict[Mode, RepositoryProtocol]]:
        return self._repos

    def insert(self, device: Device, repos: Dict[Mode, RepositoryProtocol]):
        if device.type != "mps":
            raise ValueError(f"Device type must be 'mps', got {device.type}")

        self._repos = repos


class _CUDARepos(DeviceRepos):
    _repos: IntervalTree[Dict[Mode, RepositoryProtocol]]

    def __init__(self):
        super().__init__()
        self.repos_by_capability = IntervalTree()

    @property
    def repos(
        self,
    ) -> Optional[Dict[Mode, RepositoryProtocol]]:
        capability = _find_capability()
        return self.repos_by_capability.find_smallest_interval(capability)

    def insert(self, device: Device, repos: Dict[Mode, RepositoryProtocol]):
        assert device.properties is None or isinstance(
            device.properties, CUDAProperties
        )

        min_capability = (
            0 if device.properties is None else device.properties.min_capability
        )
        max_capability = (
            sys.maxsize
            if device.properties is None
            else device.properties.max_capability
        )

        self.repos_by_capability.insert(min_capability, max_capability, repos)


class _ROCMRepos(DeviceRepos):
    _repos: IntervalTree[Dict[Mode, RepositoryProtocol]]

    def __init__(self):
        super().__init__()
        self.repos_by_capability = IntervalTree()

    @property
    def repos(
        self,
    ) -> Optional[Dict[Mode, RepositoryProtocol]]:
        capability = _find_capability()
        return self.repos_by_capability.find_smallest_interval(capability)

    def insert(self, device: Device, repos: Dict[Mode, RepositoryProtocol]):
        assert device.properties is None or isinstance(
            device.properties, ROCMProperties
        )

        min_capability = (
            0 if device.properties is None else device.properties.min_capability
        )
        max_capability = (
            sys.maxsize
            if device.properties is None
            else device.properties.max_capability
        )

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
    Mode.INFERENCE
    | Mode.TORCH_COMPILE: [
        Mode.INFERENCE | Mode.TORCH_COMPILE,
        Mode.TRAINING | Mode.TORCH_COMPILE,
        Mode.FALLBACK,
    ],
    Mode.TRAINING
    | Mode.TORCH_COMPILE: [
        Mode.TRAINING | Mode.TORCH_COMPILE,
        Mode.FALLBACK,
    ],
}


def _select_repository(
    repositories: Dict[Mode, RepositoryProtocol],
    *,
    mode: Mode,
) -> Optional[Tuple[RepositoryProtocol, Mode]]:
    # Get the fallback priority list for the requested mode
    if mode not in _MODE_FALLBACK_PRIORITY:
        raise ValueError(f"Unsupported mode: {mode}")

    fallback_modes = _MODE_FALLBACK_PRIORITY[mode]

    # Try each mode in priority order
    for fallback_mode in fallback_modes:
        if fallback_mode in repositories:
            return (repositories[fallback_mode], fallback_mode)

    return None


@lru_cache
def _find_capability() -> int:
    import torch

    major, minor = torch.cuda.get_device_capability(device=None)
    return major * 10 + minor
