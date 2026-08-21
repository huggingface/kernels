import importlib.metadata

__version__ = importlib.metadata.version("kernels")

from kernels_data import Metadata

from kernels._windows import _add_additional_dll_paths
from kernels.benchmark import Benchmark
from kernels.deps import get_kernel_dep
from kernels.hf_hub import RepoInfo
from kernels.importer import LoadedKernel, get_loaded_kernels
from kernels.install import install_kernel
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
    ROCMProperties,
    kernelize,
    register_kernel_mapping,
    replace_kernel_forward_from_hub,
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub,
    use_kernel_mapping,
    use_kernelized_func,
)
from kernels.load import (
    get_kernel,
    get_local_kernel,
    get_locked_kernel,
    has_kernel,
    load_kernel,
)
from kernels.variants import (
    VariantAccepted,
    VariantRejected,
    get_kernel_variants,
)

_add_additional_dll_paths()

__all__ = [
    "__version__",
    "Benchmark",
    "CUDAProperties",
    "Device",
    "ROCMProperties",
    "FuncRepository",
    "LayerRepository",
    "LoadedKernel",
    "LocalFuncRepository",
    "LocalLayerRepository",
    "LockedFuncRepository",
    "LockedLayerRepository",
    "Metadata",
    "Mode",
    "RepoInfo",
    "VariantAccepted",
    "VariantRejected",
    "get_kernel",
    "get_kernel_dep",
    "get_kernel_variants",
    "get_loaded_kernels",
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
    "use_kernelized_func",
]
