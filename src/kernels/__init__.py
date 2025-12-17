import importlib.metadata


__version__ = importlib.metadata.version("kernels")

from kernels.layer import (
    CUDAProperties,
    Device,
    FuncRepository,
    LayerRepository,
    LocalFuncRepository,
    LocalLayerRepository,
    LockedFuncRepository,
    LockedLayerRepository,
    Mode,
    kernelize,
    register_kernel_mapping,
    replace_kernel_forward_from_hub,
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub,
    use_kernel_mapping,
)
from kernels.utils import (
    get_kernel,
    get_local_kernel,
    get_locked_kernel,
    has_kernel,
    install_kernel,
    load_kernel,
)


from kernels._windows import _add_additional_dll_paths

_add_additional_dll_paths()

__all__ = [
    "__version__",
    "CUDAProperties",
    "Device",
    "FuncRepository",
    "LayerRepository",
    "LocalFuncRepository",
    "LocalLayerRepository",
    "LockedFuncRepository",
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
    "use_kernel_forward_from_hub",
    "use_kernel_func_from_hub",
    "use_kernel_mapping",
]
