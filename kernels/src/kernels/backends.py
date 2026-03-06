import ctypes
import platform
import warnings
from dataclasses import dataclass
from typing import Optional

from packaging.version import Version, parse

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
    pass

    @property
    def name(self) -> str:
        return "metal"

    def __str__(self) -> str:
        return "metal"


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


Backend = CANN | CPU | CUDA | Metal | ROCm | XPU


def _backend() -> Backend:
    if has_torch:
        import torch

        if torch.version.cuda is not None:
            cuda_version = parse(torch.version.cuda)
            return CUDA(version=cuda_version)
        elif torch.version.hip is not None:
            rocm_version = parse(torch.version.hip.split("-")[0])
            return ROCm(version=rocm_version)
        elif torch.backends.mps.is_available():
            return Metal()
        elif hasattr(torch.version, "xpu") and torch.version.xpu is not None:
            version = torch.version.xpu[0:6]
            return XPU(version=parse(version))
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

    raise ValueError(f"Invalid backend '{backend}', system supported backends: {', '.join(sorted(supported.keys()))}")


def _supported_backends() -> dict[str, Backend]:
    backend = _backend()
    return {"cpu": CPU(), backend.name: backend}


def _get_cuda() -> Optional[CUDA]:
    """
    Load the CUDA driver library using ctypes.

    Returns a CUDA instance if the driver library was found and loaded
    successfully, or None otherwise.
    """
    system = platform.system()

    if system == "Linux":
        lib_names = [
            "libcuda.so.1",
            "libcuda.so",
            # NixOS exposes the CUDA driver at a well-known path outside
            # of the ldconfig cache.
            "/run/opengl-driver/lib/libcuda.so",
        ]
    elif system == "Windows":
        lib_names = ["nvcuda.dll"]
    else:
        return None

    libcuda = None
    for lib_name in lib_names:
        try:
            libcuda = ctypes.CDLL(lib_name)
            # cuInit must be called before any other driver API function.
            # Passing 0 is required by the API (flags must be 0).
            result = libcuda.cuInit(0)
            if result != 0:
                warnings.warn(
                    "System has CUDA driver library, but cannot be initialized. "
                    "This usually means that no devices are visible."
                )
                return None
        except OSError:
            continue

    if libcuda is None:
        return None

    driver_version = ctypes.c_int(0)
    result = libcuda.cuDriverGetVersion(ctypes.byref(driver_version))
    if result != 0:
        warnings.warn("System has CUDA driver library, but cannot get driver version.")
        return None

    # cuDriverGetVersion encodes the version as (major * 1000 + minor * 10),
    # e.g. 12040 for CUDA 12.4.
    version_int = driver_version.value
    major = version_int // 1000
    minor = (version_int % 1000) // 10

    return CUDA(version=Version(f"{major}.{minor}"))
