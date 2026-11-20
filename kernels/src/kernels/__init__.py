import importlib.metadata

__version__ = importlib.metadata.version("kernels")

from kernels._internals._windows import _add_additional_dll_paths
from kernels._internals.benchmark import Benchmark
from kernels._internals.deps import get_kernel_dep
from kernels._internals.hf_hub import RepoInfo
from kernels._internals.importer import LoadedKernel, get_loaded_kernels
from kernels._internals.install import install_kernel
from kernels._internals.layer import (
    CUDAProperties,
    Device,
    FuncRepository,
    KernelizedFunc,
    KernelizedLayer,
    KernelizedLayers,
    KernelizeFallback,
    LayerRepository,
    LocalFuncRepository,
    LocalLayerRepository,
    LockedFuncRepository,
    LockedLayerRepository,
    Mode,
    ROCMProperties,
    get_kernelized_layers,
    kernelize,
    register_kernel_mapping,
    replace_kernel_forward_from_hub,
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub,
    use_kernel_mapping,
    use_kernelized_func,
)
from kernels._internals.load import (
    get_kernel,
    get_local_kernel,
    get_locked_kernel,
    has_kernel,
    load_kernel,
)
from kernels._internals.variants import (
    VariantAccepted,
    VariantRejected,
    get_kernel_variants,
)
from kernels._rust import Metadata

_add_additional_dll_paths()

__all__ = [
    "__version__",
    "Benchmark",
    "CUDAProperties",
    "Device",
    "KernelizeFallback",
    "KernelizedFunc",
    "KernelizedLayer",
    "KernelizedLayers",
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
    "get_kernelized_layers",
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
