import ctypes
import ctypes.util
import re
import warnings
from dataclasses import dataclass
from typing import ClassVar, Optional, Protocol, runtime_checkable

from huggingface_hub.dataclasses import strict
from packaging.version import Version

from kernels.compat import has_torch


@runtime_checkable
class Backend(Protocol):
    @property
    def name(self) -> str:
        """
        Short name of the backend, e.g. "cuda", "rocm", "cpu", etc.
        """
        ...

    @property
    def variant_str(self) -> str:
        """
        The name of the backend as used in a build variant, e.g. `cu128`
        for CUDA 12.8.
        """
        ...


@dataclass(unsafe_hash=True)
class CANN:
    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"cann(\d+)(\d+)")

    version: Version

    @property
    def name(self) -> str:
        return "cann"

    @property
    def variant_str(self) -> str:
        return f"cann{self.version.major}{self.version.minor}"

    @staticmethod
    def parse(s: str) -> "CANN":
        m = CANN._VARIANT_REGEX.fullmatch(s)
        if not m:
            raise ValueError(f"Invalid CANN variant string: {s!r}")
        return CANN(version=Version(f"{m.group(1)}.{m.group(2)}"))


@strict
@dataclass(unsafe_hash=True)
class CPU:
    @property
    def name(self) -> str:
        return "cpu"

    @property
    def variant_str(self) -> str:
        return "cpu"

    @staticmethod
    def parse(s: str) -> "CPU":
        if s != "cpu":
            raise ValueError(f"Invalid CPU variant string: {s!r}")
        return CPU()


@dataclass(unsafe_hash=True)
class CUDA:
    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"cu(\d+)(\d+)")

    version: Version

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def variant_str(self) -> str:
        return f"cu{self.version.major}{self.version.minor}"

    @staticmethod
    def parse(s: str) -> "CUDA":
        m = CUDA._VARIANT_REGEX.fullmatch(s)
        if not m:
            raise ValueError(f"Invalid CUDA variant string: {s!r}")
        return CUDA(version=Version(f"{m.group(1)}.{m.group(2)}"))


@strict
@dataclass(unsafe_hash=True)
class Metal:
    @property
    def name(self) -> str:
        return "metal"

    @property
    def variant_str(self) -> str:
        return "metal"

    @staticmethod
    def parse(s: str) -> "Metal":
        if s != "metal":
            raise ValueError(f"Invalid Metal variant string: {s!r}")
        return Metal()


@strict
@dataclass(unsafe_hash=True)
class Neuron:
    @property
    def name(self) -> str:
        return "neuron"

    @property
    def variant_str(self) -> str:
        return "neuron"

    @staticmethod
    def parse(s: str) -> "Neuron":
        if s != "neuron":
            raise ValueError(f"Invalid Neuron variant string: {s!r}")
        return Neuron()


@dataclass(unsafe_hash=True)
class ROCm:
    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"rocm(\d+)(\d+)")

    version: Version

    @property
    def name(self) -> str:
        return "rocm"

    @property
    def variant_str(self) -> str:
        return f"rocm{self.version.major}{self.version.minor}"

    @staticmethod
    def parse(s: str) -> "ROCm":
        m = ROCm._VARIANT_REGEX.fullmatch(s)
        if not m:
            raise ValueError(f"Invalid ROCm variant string: {s!r}")
        return ROCm(version=Version(f"{m.group(1)}.{m.group(2)}"))


@dataclass(unsafe_hash=True)
class XPU:
    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"xpu(\d+)(\d+)")

    version: Version

    @property
    def name(self) -> str:
        return "xpu"

    @property
    def variant_str(self) -> str:
        return f"xpu{self.version.major}{self.version.minor}"

    @staticmethod
    def parse(s: str) -> "XPU":
        m = XPU._VARIANT_REGEX.fullmatch(s)
        if not m:
            raise ValueError(f"Invalid XPU variant string: {s!r}")
        return XPU(version=Version(f"{m.group(1)}.{m.group(2)}"))


def parse_backend(s: str) -> Backend:
    """Parse a backend variant string (e.g. 'cu128', 'rocm61', 'cpu') into a Backend."""
    if s == "cpu":
        return CPU.parse(s)
    elif s == "metal":
        return Metal.parse(s)
    elif s == "neuron":
        return Neuron.parse(s)
    elif s.startswith("cu"):
        return CUDA.parse(s)
    elif s.startswith("rocm"):
        return ROCm.parse(s)
    elif s.startswith("xpu"):
        return XPU.parse(s)
    elif s.startswith("cann"):
        return CANN.parse(s)
    else:
        raise ValueError(f"Unknown backend variant string: {s!r}")


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

    raise ValueError(f"Invalid backend '{backend}', system supported backends: {', '.join(sorted(supported.keys()))}")


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
        warnings.warn("System has CUDA runtime library, but cannot get runtime version.")
        return None

    # cudaRuntimeGetVersion encodes the version as (major * 1000 + minor * 10).
    version_int = runtime_version.value
    major = version_int // 1000
    minor = (version_int % 1000) // 10

    return CUDA(version=Version(f"{major}.{minor}"))
