import importlib.metadata

__version__ = importlib.metadata.version("kernels")

from kernels.layer import Device, CUDAProperties
from kernels.layer import kernelize, register_kernel_mapping, use_kernel_mapping
from kernels.layer import Mode
from kernels.layer import (
    LayerRepository,
    LocalLayerRepository,
    LockedLayerRepository,
    replace_kernel_forward_from_hub,
    use_kernel_forward_from_hub,
)
from kernels.utils import (
    get_kernel,
    get_local_kernel,
    get_locked_kernel,
    has_kernel,
    install_kernel,
    load_kernel,
)

__all__ = [
    "__version__",
    "CUDAProperties",
    "Device",
    "LayerRepository",
    "LocalLayerRepository",
    "LockedLayerRepository",
    "Mode",
    "get_kernel",
    "get_local_kernel",
    "get_locked_kernel",
    "has_kernel",
    "install_kernel",
    "kernelize",
    "load_kernel",
    "register_kernel_mapping",
    "replace_kernel_forward_from_hub",
    "replace_kernel_func_from_hub",
    "use_kernel_forward_from_hub",
    "use_kernel_func_from_hub",
    "use_kernel_mapping",
]
