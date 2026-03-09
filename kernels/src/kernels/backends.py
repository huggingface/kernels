import ctypes
import ctypes.util
import warnings
from dataclasses import dataclass
from typing import Optional

from packaging.version import Version

from kernels.compat import has_torch


@dataclass
class CANN:
    version: Version

    def __str__(self) -> str:
        return f"cann{self.version.major}{self.version.minor}"

    @property
    def name(self) -> str:
        return "cann"


@dataclass
class CPU:
    @property
    def name(self) -> str:
        return "cpu"

    def __str__(self) -> str:
        return "cpu"


@dataclass
class CUDA:
    version: Version

    @property
    def name(self) -> str:
        return "cuda"

    def __str__(self) -> str:
        return f"cu{self.version.major}{self.version.minor}"


@dataclass
class Metal:
    @property
    def name(self) -> str:
        return "metal"

    def __str__(self) -> str:
        return "metal"


@dataclass
class Neuron:
    @property
    def name(self) -> str:
        return "neuron"

    def __str__(self) -> str:
        return "neuron"


@dataclass
class ROCm:
    version: Version

    @property
    def name(self) -> str:
        return "rocm"

    def __str__(self) -> str:
        return f"rocm{self.version.major}{self.version.minor}"


@dataclass
class XPU:
    version: Version

    @property
    def name(self) -> str:
        return "xpu"

    def __str__(self) -> str:
        return f"xpu{self.version.major}{self.version.minor}"


Backend = CANN | CPU | CUDA | Metal | Neuron | ROCm | XPU


def _backend() -> Backend:
    if has_torch:
        import torch

        if hasattr(torch, "neuron"):
            # Needs to be sorted before specific Torch builds, since Neuron
            # extension can be loaded into e.g. CUDA Torch builds.
            return Neuron()
        elif torch.version.cuda is not None:
            cuda_version = Version(torch.version.cuda)
            return CUDA(version=cuda_version)
        elif torch.version.hip is not None:
            rocm_version = Version(torch.version.hip.split("-")[0])
            return ROCm(version=rocm_version)
        elif torch.backends.mps.is_available():
            return Metal()
        elif hasattr(torch.version, "xpu") and torch.version.xpu is not None:
            version = f"{torch.version.xpu[0:4]}.{torch.version.xpu[5:6]}"
            return XPU(version=Version(version))
        elif _get_torch_privateuse_backend_name() == "npu":
            from torch_npu.utils.collect_env import get_cann_version  # type: ignore[import-not-found]

            cann_major, cann_minor = get_cann_version()[0], get_cann_version()[2]
            return CANN(version=Version(f"{cann_major}.{cann_minor}"))
        else:
            return CPU()
    else:
        cuda = _get_cuda()
        if cuda is not None:
            return cuda

        return CPU()


def _get_torch_privateuse_backend_name() -> str | None:
    import torch

    if hasattr(torch._C, "_get_privateuse1_backend_name"):
        return torch._C._get_privateuse1_backend_name()
    return None


def _select_backend(backend: str | None) -> Backend:
    if backend is None:
        return _backend()

    supported = _supported_backends()
    if backend in supported:
        return supported[backend]

    raise ValueError(
        f"Invalid backend '{backend}', system supported backends: {', '.join(sorted(supported.keys()))}"
    )


def _supported_backends() -> dict[str, Backend]:
    backend = _backend()
    return {"cpu": CPU(), backend.name: backend}


def _get_cuda() -> Optional[CUDA]:
    """
    Get CUDA runtime library information.
    """
    lib_name = ctypes.util.find_library("cudart")
    if lib_name is None:
        return None

    try:
        libcudart = ctypes.CDLL(lib_name)
    except OSError:
        return None

    runtime_version = ctypes.c_int(0)
    result = libcudart.cudaRuntimeGetVersion(ctypes.byref(runtime_version))
    if result != 0:
        warnings.warn(
            "System has CUDA runtime library, but cannot get runtime version."
        )
        return None

    # cudaRuntimeGetVersion encodes the version as (major * 1000 + minor * 10).
    version_int = runtime_version.value
    major = version_int // 1000
    minor = (version_int % 1000) // 10

    return CUDA(version=Version(f"{major}.{minor}"))
