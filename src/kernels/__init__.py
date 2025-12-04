import os
import platform
import importlib.metadata


if platform.system() == "Windows":
    # Add Intel oneAPI directories for XPU support
    _oneapi_paths = [
        r"C:\Program Files (x86)\Intel\oneAPI\dnnl\latest\bin",
    ]

    for _path in _oneapi_paths:
        if os.path.exists(_path):
            try:
                os.add_dll_directory(_path)
            except Exception:
                pass  # Ignore if already added or permission issues

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
