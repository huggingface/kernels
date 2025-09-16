from __future__ import annotations

import functools
import inspect
import logging
import os
import sys
import warnings
from abc import ABC, abstractmethod
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from enum import Flag, auto
from functools import lru_cache
from pathlib import Path
from types import MethodType, ModuleType
from typing import (
    TYPE_CHECKING,
    Dict,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
)

from ._interval_tree import IntervalTree
from ._versions import select_revision_or_version
from .utils import (
    _get_caller_locked_kernel,
    _get_locked_kernel,
    get_kernel,
    get_local_kernel,
)

if TYPE_CHECKING:
    import torch
    from torch import nn

_DISABLE_KERNEL_MAPPING: bool = bool(int(os.environ.get("DISABLE_KERNEL_MAPPING", "0")))


class Mode(Flag):
    """
    Kernelize mode

    The `Mode` flag is used by [`kernelize`] to select kernels for the given mode. Mappings can be registered for
    specific modes.

    Attributes:
        INFERENCE: The kernel is used for inference.
        TRAINING: The kernel is used for training.
        TORCH_COMPILE: The kernel is used with `torch.compile`.
        FALLBACK: In a kernel mapping, this kernel is used when no other mode matches.

    Note:
        Different modes can be combined. For instance, `INFERENCE | TORCH_COMPILE` should be used for layers that
        are used for inference *with* `torch.compile`.

    """

    _NONE = 0
    FALLBACK = auto()
    TRAINING = auto()
    INFERENCE = auto()
    TORCH_COMPILE = auto()

    def __or__(self, other: Mode) -> Mode:
        union = super().__or__(other)

        if Mode.INFERENCE in union and Mode.TRAINING in union:
            raise ValueError("Mode.INFERENCE and Mode.TRAINING are mutually exclusive.")

        if Mode.FALLBACK in union and union != Mode.FALLBACK:
            raise ValueError("Mode.FALLBACK cannot be combined with other modes.")

        return union


@dataclass(frozen=True)
class Device:
    """
    Represents a compute device with optional properties.

    This class encapsulates device information including device type and optional device-specific properties
    like CUDA capabilities.

    Args:
        type (`str`):
            The device type (e.g., "cuda", "mps", "rocm", "xpu", "npu").
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

        # NPU device (e.g., Huawei Ascend Atlas A2)
        npu_device = Device(type="npu")
        ```
    """

    type: str
    properties: Optional[CUDAProperties] = None

    def __post_init__(self):
        if self.properties is not None and isinstance(self.properties, CUDAProperties):
            if self.type != "cuda":
                raise ValueError("CUDAProperties is only supported for 'cuda' devices.")

    def create_repo(self) -> _DeviceRepos:
        """Create an appropriate repository set for this device type."""
        if self.type == "cuda":
            return _CUDARepos()
        elif self.type == "rocm":
            return _ROCMRepos()
        elif self.type == "mps":
            return _MPSRepos()
        elif self.type == "xpu":
            return _XPURepos()
        elif self.type == "npu":
            return _NPURepos()
        else:
            raise ValueError(f"Unknown device type: {self.type}")

    def __eq__(self, other):
        if not isinstance(other, Device):
            return NotImplemented
        return self.type == other.type and self.properties == other.properties

    def __hash__(self):
        return hash((self.type, self.properties))


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


class LayerRepositoryProtocol(Protocol):
    @property
    def layer_name(self) -> str: ...

    def load(self) -> ModuleType: ...


class LayerRepository:
    """
    Repository and name of a layer for kernel mapping.

    Args:
        repo_id (`str`):
            The Hub repository containing the layer.
        layer_name (`str`):
            The name of the layer within the kernel repository.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific revision (branch, tag, or commit) to download. Cannot be used together with `version`.
        version (`str`, *optional*):
            The kernel version to download. This can be a Python version specifier, such as `">=1.0.0,<2.0.0"`.
            Cannot be used together with `revision`.

    Example:
        ```python
        from kernels import LayerRepository

        # Reference a specific layer by revision
        layer_repo = LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
        )

        # Reference a layer by version constraint
        layer_repo_versioned = LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
            version=">=0.0.3,<0.1"
        )
        ```
    """

    def __init__(
        self,
        repo_id: str,
        *,
        layer_name: str,
        revision: Optional[str] = None,
        version: Optional[str] = None,
    ):
        if revision is not None and version is not None:
            raise ValueError(
                "Either a revision or a version must be specified, not both."
            )

        self._repo_id = repo_id
        self.layer_name = layer_name

        # We are going to resolve these lazily, since we do not want
        # to do a network request for every registered LayerRepository.
        self._revision = revision
        self._version = version

    @functools.lru_cache()
    def _resolve_revision(self) -> str:
        return select_revision_or_version(
            repo_id=self._repo_id, revision=self._revision, version=self._version
        )

    def load(self) -> ModuleType:
        return get_kernel(self._repo_id, revision=self._resolve_revision())

    def __eq__(self, other):
        return (
            isinstance(other, LayerRepository)
            and self.layer_name == other.layer_name
            and self._repo_id == other._repo_id
            and self._revision == other._revision
            and self._version == other._version
        )

    def __hash__(self):
        return hash((self.layer_name, self._repo_id, self._revision, self._version))

    def __str__(self) -> str:
        return f"`{self._repo_id}` (revision: {self._resolve_revision()}) for layer `{self.layer_name}`"


class LocalLayerRepository:
    """
    Repository from a local directory for kernel mapping.

    Args:
        repo_path (`Path`):
            The local repository containing the layer.
        package_name (`str`):
            Package name of the kernel.
        layer_name (`str`):
            The name of the layer within the kernel repository.

    Example:
        ```python
        from pathlib import Path

        from kernels import LocalLayerRepository

        # Reference a specific layer by revision
        layer_repo = LocalLayerRepository(
            repo_path=Path("/home/daniel/kernels/activation"),
            package_name="activation",
            layer_name="SiluAndMul",
        )
        ```
    """

    def __init__(
        self,
        repo_path: Path,
        *,
        package_name: str,
        layer_name: str,
    ):
        self._repo_path = repo_path
        self._package_name = package_name
        self.layer_name = layer_name

    def load(self) -> ModuleType:
        return get_local_kernel(self._repo_path, self._package_name)

    def __eq__(self, other):
        return (
            isinstance(other, LocalLayerRepository)
            and self.layer_name == other.layer_name
            and self._repo_path == other._repo_path
            and self._package_name == other._package_name
        )

    def __hash__(self):
        return hash((self.layer_name, self._repo_path, self._package_name))

    def __str__(self) -> str:
        return f"`{self._repo_path}` (package: {self._package_name}) for layer `{self.layer_name}`"


class LockedLayerRepository:
    """
    Repository and name of a layer.

    In contrast to `LayerRepository`, this class uses repositories that
    are locked inside a project.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        lockfile: Optional[Path] = None,
        layer_name: str,
    ):
        """
        Construct a layer repository.

        Args:
            repo_id (`str`): The Hub repository containing the layer.
        """
        self._repo_id = repo_id
        self._lockfile = lockfile
        self.layer_name = layer_name

    @functools.lru_cache()
    def _resolve_revision(self) -> str:
        if self._lockfile is None:
            locked_sha = _get_caller_locked_kernel(self._repo_id)
        else:
            with open(self._lockfile, "r") as f:
                locked_sha = _get_locked_kernel(self._repo_id, f.read())

        if locked_sha is None:
            raise ValueError(f"Kernel `{self._repo_id}` is not locked")

        return locked_sha

    def load(self) -> ModuleType:
        return get_kernel(repo_id=self._repo_id, revision=self._resolve_revision())

    def __eq__(self, other):
        return (
            isinstance(other, LockedLayerRepository)
            and self.layer_name == other.layer_name
            and self._repo_id == other._repo_id
        )

    def __hash__(self):
        return hash((self.layer_name, self._repo_id))

    def __str__(self) -> str:
        return f"`{self._repo_id}` (revision: {self._resolve_revision()}) for layer `{self.layer_name}`"


_CACHED_LAYER: Dict[LayerRepositoryProtocol, Type["nn.Module"]] = {}


class _DeviceRepos(ABC):
    """
    Device-specific kernel layer repositories.
    """

    @property
    @abstractmethod
    def repos(
        self,
    ) -> Optional[Dict[Mode, LayerRepositoryProtocol]]: ...

    @abstractmethod
    def insert(self, device: Device, repos: Dict[Mode, LayerRepositoryProtocol]):
        """
        Insert a repository for a specific device and mode.
        """
        ...


class _XPURepos(_DeviceRepos):
    _repos: Dict[Mode, LayerRepositoryProtocol]

    def __init__(self):
        super().__init__()
        self._repos = {}

    @property
    def repos(
        self,
    ) -> Optional[Dict[Mode, LayerRepositoryProtocol]]:
        return self._repos

    def insert(self, device: Device, repos: Dict[Mode, LayerRepositoryProtocol]):
        if device.type != "xpu":
            raise ValueError(f"Device type must be 'xpu', got {device.type}")

        self._repos = repos


class _NPURepos(_DeviceRepos):
    _repos: Dict[Mode, LayerRepositoryProtocol]

    def __init__(self):
        super().__init__()
        self._repos = {}

    @property
    def repos(
        self,
    ) -> Optional[Dict[Mode, LayerRepositoryProtocol]]:
        return self._repos

    def insert(self, device: Device, repos: Dict[Mode, LayerRepositoryProtocol]):
        if device.type != "npu":
            raise ValueError(f"Device type must be 'npu', got {device.type}")

        self._repos = repos


class _MPSRepos(_DeviceRepos):
    _repos: Dict[Mode, LayerRepositoryProtocol]

    def __init__(self):
        super().__init__()
        self._repos = {}

    @property
    def repos(
        self,
    ) -> Optional[Dict[Mode, LayerRepositoryProtocol]]:
        return self._repos

    def insert(self, device: Device, repos: Dict[Mode, LayerRepositoryProtocol]):
        if device.type != "mps":
            raise ValueError(f"Device type must be 'mps', got {device.type}")

        self._repos = repos


class _CUDARepos(_DeviceRepos):
    _repos: IntervalTree[Dict[Mode, LayerRepositoryProtocol]]

    def __init__(self):
        super().__init__()
        self.repos_by_capability = IntervalTree()

    @property
    def repos(
        self,
    ) -> Optional[Dict[Mode, LayerRepositoryProtocol]]:
        capability = _find_capability()
        return self.repos_by_capability.find_smallest_interval(capability)

    def insert(self, device: Device, repos: Dict[Mode, LayerRepositoryProtocol]):
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


class _ROCMRepos(_DeviceRepos):
    _repos: IntervalTree[Dict[Mode, LayerRepositoryProtocol]]

    def __init__(self):
        super().__init__()
        self.repos_by_capability = IntervalTree()

    @property
    def repos(
        self,
    ) -> Optional[Dict[Mode, LayerRepositoryProtocol]]:
        capability = _find_capability()
        return self.repos_by_capability.find_smallest_interval(capability)

    def insert(self, device: Device, repos: Dict[Mode, LayerRepositoryProtocol]):
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


def _validate_device_type(device_type: str) -> None:
    """Validate that the device type is supported."""
    supported_devices = {"cuda", "rocm", "mps", "xpu", "npu"}
    if device_type not in supported_devices:
        raise ValueError(
            f"Unsupported device type '{device_type}'. Supported device types are: {', '.join(sorted(supported_devices))}"
        )


_KERNEL_MAPPING: ContextVar[Dict[str, Dict[str, _DeviceRepos]]] = ContextVar(
    "_KERNEL_MAPPING", default={}
)


def use_kernel_mapping(
    mapping: Dict[
        str,
        Dict[
            Union[Device, str],
            Union[LayerRepositoryProtocol, Dict[Mode, LayerRepositoryProtocol]],
        ],
    ],
    *,
    inherit_mapping: bool = True,
):
    """
    Context manager that sets a kernel mapping for the duration of the context.

    This function allows temporary kernel mappings to be applied within a specific context, enabling different
    kernel configurations for different parts of your code.

    Args:
        mapping (`Dict[str, Dict[Union[Device, str], Union[LayerRepositoryProtocol, Dict[Mode, LayerRepositoryProtocol]]]]`):
            The kernel mapping to apply. Maps layer names to device-specific kernel configurations.
        inherit_mapping (`bool`, *optional*, defaults to `True`):
            When `True`, the current mapping will be extended by `mapping` inside the context. When `False`,
            only `mapping` is used inside the context.

    Returns:
        Context manager that handles the temporary kernel mapping.

    Example:
        ```python
        import torch
        import torch.nn as nn
        from torch.nn import functional as F

        from kernels import use_kernel_forward_from_hub
        from kernels import use_kernel_mapping, LayerRepository, Device
        from kernels import Mode, kernelize

        # Define a mapping
        mapping = {
            "SiluAndMul": {
                "cuda": LayerRepository(
                    repo_id="kernels-community/activation",
                    layer_name="SiluAndMul",
                )
            }
        }

        @use_kernel_forward_from_hub("SiluAndMul")
        class SiluAndMul(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                d = x.shape[-1] // 2
                return F.silu(x[..., :d]) * x[..., d:]

        model = SiluAndMul()

        # Use the mapping for the duration of the context.
        with use_kernel_mapping(mapping):
            # kernelize uses the temporary mapping
            model = kernelize(model, mode=Mode.TRAINING | Mode.TORCH_COMPILE, device="cuda")

        # Outside the context, original mappings are restored
        ```
    """

    class ContextManager:
        def __enter__(self):
            # Mappings always stack on previous mappings.
            if inherit_mapping:
                self.token = _KERNEL_MAPPING.set(deepcopy(_KERNEL_MAPPING.get()))
            else:
                self.token = _KERNEL_MAPPING.set({})
            register_kernel_mapping(mapping)

        def __exit__(self, exc_type, exc_value, traceback):
            _KERNEL_MAPPING.reset(self.token)

    return ContextManager()


def register_kernel_mapping(
    mapping: Dict[
        str,
        Dict[
            Union[Device, str],
            Union[LayerRepositoryProtocol, Dict[Mode, LayerRepositoryProtocol]],
        ],
    ],
    inherit_mapping: bool = True,
):
    """
    Register a global mapping between layer names and their corresponding kernel implementations.

    This function allows you to register a mapping between a layer name and the corresponding kernel(s) to use,
    depending on the device and mode. This should be used in conjunction with [`kernelize`].

    Args:
        mapping (`Dict[str, Dict[Union[Device, str], Union[LayerRepositoryProtocol, Dict[Mode, LayerRepositoryProtocol]]]]`):
            The kernel mapping to register globally. Maps layer names to device-specific kernels.
            The mapping can specify different kernels for different modes (training, inference, etc.).
        inherit_mapping (`bool`, *optional*, defaults to `True`):
            When `True`, the current mapping will be extended by `mapping`. When `False`, the existing mappings
            are erased before adding `mapping`.

    Example:
        ```python
        from kernels import LayerRepository, register_kernel_mapping, Mode

        # Simple mapping for a single kernel per device
        kernel_layer_mapping = {
            "LlamaRMSNorm": {
                "cuda": LayerRepository(
                    repo_id="kernels-community/activation",
                    layer_name="RmsNorm",
                    revision="layers",
                ),
            },
        }
        register_kernel_mapping(kernel_layer_mapping)

        # Advanced mapping with mode-specific kernels
        advanced_mapping = {
            "MultiHeadAttention": {
                "cuda": {
                    Mode.TRAINING: LayerRepository(
                        repo_id="username/training-kernels",
                        layer_name="TrainingAttention"
                    ),
                    Mode.INFERENCE: LayerRepository(
                        repo_id="username/inference-kernels",
                        layer_name="FastAttention"
                    ),
                }
            }
        }
        register_kernel_mapping(advanced_mapping)
        ```
    """
    if not inherit_mapping:
        _KERNEL_MAPPING.set({})

    # Merge with existing mappings.
    for new_kernel, new_device_repos in mapping.items():
        device_repo = _KERNEL_MAPPING.get().setdefault(new_kernel, {})
        for new_device, new_repo in new_device_repos.items():
            device = (
                Device(type=new_device) if isinstance(new_device, str) else new_device
            )

            if isinstance(new_repo, dict):
                kernel_options = new_repo
            else:
                kernel_options = {Mode.FALLBACK: new_repo}

            feature_repos = device_repo.setdefault(device.type, device.create_repo())
            feature_repos.insert(device, kernel_options)


def replace_kernel_forward_from_hub(
    cls,
    layer_name: str,
):
    """
    Function that prepares a layer class to use kernels from the Hugging Face Hub.

    It is recommended to use [`use_kernel_forward_from_hub`] decorator instead.
    This function should only be used as a last resort to extend third-party layers,
    it is inherently fragile since the member variables and `forward` signature
    of usch a layer can change.

    Example:
        ```python
        from kernels import replace_kernel_forward_from_hub
        import torch.nn as nn

        replace_kernel_forward_from_hub(nn.LayerNorm, "LayerNorm")
        ```
    """
    cls.kernel_layer_name = layer_name


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
    repositories: Dict[Mode, LayerRepositoryProtocol],
    *,
    mode: Mode,
) -> Optional[Tuple[LayerRepositoryProtocol, Mode]]:
    # Get the fallback priority list for the requested mode
    if mode not in _MODE_FALLBACK_PRIORITY:
        raise ValueError(f"Unsupported mode: {mode}")

    fallback_modes = _MODE_FALLBACK_PRIORITY[mode]

    # Try each mode in priority order
    for fallback_mode in fallback_modes:
        if fallback_mode in repositories:
            return (repositories[fallback_mode], fallback_mode)

    return None


def kernelize(
    model: "nn.Module",
    *,
    mode: Mode,
    device: Optional[Union[str, "torch.device"]] = None,
    use_fallback: bool = True,
):
    """
    Replace layer forward methods with optimized kernel implementations.

    This function iterates over all modules in the model and replaces the `forward` method of extensible layers
    for which kernels are registered using [`register_kernel_mapping`] or [`use_kernel_mapping`].

    Args:
        model (`nn.Module`):
            The PyTorch model to kernelize.
        mode ([`Mode`]): The mode that the kernel is going to be used in. For example,
            `Mode.TRAINING | Mode.TORCH_COMPILE` kernelizes the model for training with
            `torch.compile`.
        device (`Union[str, torch.device]`, *optional*):
            The device type to load kernels for. Supported device types are: "cuda", "mps", "rocm", "xpu", "npu".
            The device type will be inferred from the model parameters when not provided.
        use_fallback (`bool`, *optional*, defaults to `True`):
            Whether to use the original forward method of modules when no compatible kernel could be found.
            If set to `False`, an exception will be raised in such cases.

    Returns:
        `nn.Module`: The kernelized model with optimized kernel implementations.

    Example:
        ```python
        import torch
        import torch.nn as nn

        from kernels import kernelize, Mode, register_kernel_mapping, LayerRepository
        from kernels import use_kernel_forward_from_hub

        @use_kernel_forward_from_hub("SiluAndMul")
        class SiluAndMul(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                d = x.shape[-1] // 2
                return F.silu(x[..., :d]) * x[..., d:]

        mapping = {
            "SiluAndMul": {
                "cuda": LayerRepository(
                    repo_id="kernels-community/activation",
                    layer_name="SiluAndMul",
                )
            }
        }
        register_kernel_mapping(mapping)

        # Create and kernelize a model
        model = nn.Sequential(
            nn.Linear(1024, 2048, device="cuda"),
            SiluAndMul(),
        )

        # Kernelize for inference
        kernelized_model = kernelize(model, mode=Mode.TRAINING | Mode.TORCH_COMPILE)
        ```
    """

    if mode == Mode.FALLBACK:
        raise ValueError("Mode.FALLBACK can only be used to register kernel mappings.")

    # Type check ignored because this causes a false negative on Python < 3.11.
    # Looks similar to: https://github.com/python/mypy/issues/9642
    # Remove once we start doing typing checks on >= 3.11.
    if Mode.INFERENCE not in mode and Mode.TRAINING not in mode:  # type: ignore[operator]
        raise ValueError("kernelize mode must contain Mode.INFERENCE or Mode.TRAINING.")

    if device is None:
        device_type = _find_device(model)
    elif isinstance(device, str):
        _validate_device_type(device)
        device_type = Device(type=device)
    else:
        device_type = Device(device.type)

    assert isinstance(device_type, Device)

    for _, module in model.named_modules():
        module_class = type(module)
        if not hasattr(module_class, "kernel_layer_name"):
            continue
        layer_name = module_class.kernel_layer_name

        if _DISABLE_KERNEL_MAPPING:
            _replace_forward(module, module_class)
            continue

        kernel = _KERNEL_MAPPING.get().get(str(layer_name))

        if kernel is None:
            warnings.warn(
                "\n"
                f"No kernel mapping found for layer `{layer_name}`. "
                f"Check if the layer name matches one of the kernels in the mapping or add the kernel "
                f"you want to use to the mapping. Defaulting to original forward implementation."
            )
            if not use_fallback:
                raise ValueError(f"No layer mapping for `{layer_name}`")
            _replace_forward(module, module_class)
            continue

        # Get kernel options for the device
        property_repos = kernel.get(device_type.type)

        if property_repos is None:
            if not use_fallback:
                raise ValueError(
                    f"No layer mapping for `{layer_name}` with device type `{device_type}`"
                )
            _replace_forward(module, module_class)
            continue

        repos = property_repos.repos

        if repos is None:
            if not use_fallback:
                raise ValueError(
                    f"No layer mapping for `{layer_name}` device `{device_type}` with the right properties"
                )
            _replace_forward(module, module_class)
            continue

        repo_with_mode = _select_repository(
            repos,
            mode=mode,
        )

        if repo_with_mode is None:
            if not use_fallback:
                raise ValueError(
                    f"No repository for `{layer_name}` for configuration mode={mode}"
                )
            _replace_forward(module, module_class)
            continue

        repo, repo_mode = repo_with_mode

        logging.info(f"Using layer `{repo.layer_name}` from repo {repo}")
        logging.debug(f"kernelize mode: {mode}, repo mode: {repo_mode}")

        layer = _get_layer_memoize(repo, module_class)

        # Ideally we would do validation on the mapping where we check that
        # e.g. if a repo class is registered for TRAINING | TORCH_COMPILE,
        # the actual layer is compatible with that. Unfortunately, this would
        # mean that we have to pre-download everything.
        _validate_layer_has_mode(
            layer_name=layer_name, module=layer, repo=repo, repo_mode=repo_mode
        )

        _conditionally_replace_forward(
            module=module,
            layer=layer,
            mode=mode,
            use_fallback=use_fallback,
        )

    return model


def use_kernel_forward_from_hub(layer_name: str):
    """
    Decorator factory that makes a layer extensible using the specified layer name.

    This is a decorator factory that returns a decorator which prepares a layer class to use kernels from the
    Hugging Face Hub.

    Args:
        layer_name (`str`):
            The name of the layer to use for kernel lookup in registered mappings.

    Returns:
        `Callable`: A decorator function that can be applied to layer classes.

    Example:
        ```python
        import torch
        import torch.nn as nn

        from kernels import use_kernel_forward_from_hub
        from kernels import Mode, kernelize

        @use_kernel_forward_from_hub("MyCustomLayer")
        class MyCustomLayer(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, x: torch.Tensor):
                # original implementation
                return x

        model = MyCustomLayer(768)

        # The layer can now be kernelized:
        # model = kernelize(model, mode=Mode.TRAINING | Mode.TORCH_COMPILE, device="cuda")
        ```
    """

    def decorator(cls):
        replace_kernel_forward_from_hub(cls, layer_name)
        return cls

    return decorator


def _get_kernel_layer(repo: LayerRepositoryProtocol) -> Type["nn.Module"]:
    """Get a layer from a kernel."""

    kernel = repo.load()

    if getattr(kernel, "layers", None) is None:
        raise ValueError(f"Kernel repo {repo} does not define any layers.")

    layer = getattr(kernel.layers, repo.layer_name, None)
    if layer is None:
        raise ValueError(f"Layer `{repo.layer_name}` not found in kernel repo {repo}.")
    return layer


def _validate_layer(*, check_cls, cls):
    import torch.nn as nn

    # The layer must have at least have the following properties: (1) it
    # must be stateless; (2) the forward signature should correspond to
    # the signature it is replacing; (3) forward should not call other
    # methods.

    if not issubclass(cls, nn.Module):
        raise TypeError(f"Layer `{cls}` is not a Torch layer.")

    # We verify statelessness by checking that the does not have its own
    # constructor (since the constructor could add member variables)...
    if cls.__init__ is not nn.Module.__init__:
        raise TypeError("Layer must not override nn.Module constructor.")

    # ... or predefined member variables.
    torch_module_members = {name for name, _ in inspect.getmembers(nn.Module)}
    cls_members = {name for name, _ in inspect.getmembers(cls)}
    difference = cls_members - torch_module_members
    # verify if : difference ⊄ {"can_torch_compile", "has_backward"}
    if not difference <= {"can_torch_compile", "has_backward"}:
        raise TypeError("Layer must not contain additional members.")

    # Check whether the forward signatures are similar.
    params = inspect.signature(cls.forward).parameters
    ref_params = inspect.signature(check_cls.forward).parameters

    if len(params) != len(ref_params):
        raise TypeError(
            "Forward signature does not match: different number of arguments."
        )

    for param, ref_param in zip(params.values(), ref_params.values()):
        if param.kind != ref_param.kind:
            raise TypeError(
                f"Forward signature does not match: different kind of arguments ({param} ({param.kind}) and {ref_param} ({ref_param.kind})"
            )


def _is_cuda_platform():
    import torch

    return torch.version.cuda is not None


def _is_rocm_platform():
    import torch

    return torch.version.hip is not None


def _find_device(model: "nn.Module") -> Device:
    try:
        param = next(model.parameters())
    except StopIteration:
        raise ValueError(
            "Cannot determine model device, provide as `device` argument to `kernelize`."
        )

    dev_type = param.device.type
    if dev_type == "cuda":
        # Refine based on actual platform
        if _is_rocm_platform():
            return Device(type="rocm")
        elif _is_cuda_platform():
            return Device(type="cuda")

    return Device(type=dev_type)


@lru_cache
def _find_capability() -> int:
    import torch

    major, minor = torch.cuda.get_device_capability(device=None)
    return major * 10 + minor


def _conditionally_replace_forward(
    *,
    module: "nn.Module",
    layer: Type["nn.Module"],
    mode: Mode,
    use_fallback: bool,
):
    module_class = type(module)

    # Switch to fallback if the mode is not supported by the layer.
    # Note that this is useful even after _validate_layer_has_mode because
    # layers registered with the FALLBACK mode never get rejected by
    # _validate_layer_has_mode. For such layers, we want to fall back in
    # case the layer does not support the given mode.
    needs_fallback_for_compile = Mode.TORCH_COMPILE in mode and not getattr(
        layer, "can_torch_compile", False
    )
    needs_fallback_for_backward = Mode.TRAINING in mode and not getattr(
        layer, "has_backward", True
    )

    if needs_fallback_for_compile or needs_fallback_for_backward:
        if use_fallback:
            if needs_fallback_for_compile:
                logging.info("Layer does not support torch.compile, using fallback")
            if needs_fallback_for_backward:
                logging.info("Layer does not support backward, using fallback")
            _replace_forward(module, module_class)
        else:
            raise ValueError(f"Available kernel does not support mode: {mode}")
    else:
        _replace_forward(module, layer)


def _replace_forward(module: "nn.Module", layer: Type["nn.Module"]):
    module.forward = MethodType(layer.forward, module)  # type: ignore[method-assign]


def _validate_layer_has_mode(
    *,
    layer_name: str,
    module: Type["nn.Module"],
    repo: LayerRepositoryProtocol,
    repo_mode: Mode,
):
    """
    Check that a repository supports the mode that it was registered for.
    """

    if Mode.TRAINING in repo_mode and not getattr(module, "has_backward", True):
        raise ValueError(
            f"Layer `{repo.layer_name}` from repo {repo} does not support backward.\n"
            f"Was registered for `{layer_name}` with mode `{repo_mode}`"
        )

    if Mode.TORCH_COMPILE in repo_mode and not getattr(
        module, "can_torch_compile", False
    ):
        raise ValueError(
            f"Layer `{repo.layer_name}` from repo {repo} does not support torch.compile.\n"
            f"Was registered for `{layer_name}` with mode `{repo_mode}`"
        )

    return True


def _get_layer_memoize(
    repo: LayerRepositoryProtocol, module_class: Type["nn.Module"]
) -> Type["nn.Module"]:
    layer = _CACHED_LAYER.get(repo, None)
    if layer is not None:
        return layer

    layer = _get_kernel_layer(repo)
    _validate_layer(check_cls=module_class, cls=layer)
    _CACHED_LAYER[repo] = layer

    return layer
