import ctypes
import ctypes.util
import logging
import re
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Protocol, runtime_checkable

from huggingface_hub.dataclasses import strict
from packaging.version import Version

from kernels.compat import has_torch

logger = logging.getLogger(__name__)


def _parse_version(legacy_regex: re.Pattern, new_regex: re.Pattern, s: str, legacy: bool) -> Version | None:
    """
    Parse a versioned variant string with the regex for the legacy or the new
    variant string format. Both regexes capture the major and minor version.

    Legacy regexes should parse non-dotted versions like `rocm72`, whereas
    new regexes should parse dotted versions like `rocm-7.2`.
    """
    m = (legacy_regex if legacy else new_regex).fullmatch(s)
    if not m:
        return None
    return Version(f"{m.group(1)}.{m.group(2)}")


def _format_version(legacy_prefix: str, new_prefix: str, version: Version, legacy: bool) -> str:
    """
    Format a version for a variant string: `{legacy_prefix}{major}{minor}`
    in the legacy format (e.g. `cu128`) and `{new_prefix}{major}.{minor}` in
    the new format (e.g. `cuda-12.8`).
    """
    if legacy:
        return f"{legacy_prefix}{version.major}{version.minor}"
    return f"{new_prefix}{version.major}.{version.minor}"


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
        The name of the backend as used in a build variant, e.g. `cuda-12.8`
        for CUDA 12.8 (`cu128` in the legacy variant format).
        """
        ...


@dataclass(unsafe_hash=True)
class CANN:
    # Legacy variant string: `cannxy`, new variant string: `cann-x.y`.
    _LEGACY_VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"cann(\d+)(\d+)")
    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"cann-(\d+)\.(\d+)")

    version: Version
    legacy: bool = field(default=True, compare=False)

    @property
    def name(self) -> str:
        return "cann"

    @property
    def variant_str(self) -> str:
        return _format_version("cann", "cann-", self.version, self.legacy)

    @staticmethod
    def parse(s: str, *, legacy: bool = True) -> "CANN":
        version = _parse_version(CANN._LEGACY_VARIANT_REGEX, CANN._VARIANT_REGEX, s, legacy)
        if version is None:
            raise ValueError(f"Invalid CANN variant string: {s!r}")
        return CANN(version=version, legacy=legacy)


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
    # Legacy variant string: `cuxy`, new variant string: `cuda-x.y`.
    _LEGACY_VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"cu(\d+)(\d+)")
    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"cuda-(\d+)\.(\d+)")

    version: Version
    legacy: bool = field(default=True, compare=False)

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def variant_str(self) -> str:
        return _format_version("cu", "cuda-", self.version, self.legacy)

    @staticmethod
    def parse(s: str, *, legacy: bool = True) -> "CUDA":
        version = _parse_version(CUDA._LEGACY_VARIANT_REGEX, CUDA._VARIANT_REGEX, s, legacy)
        if version is None:
            raise ValueError(f"Invalid CUDA variant string: {s!r}")
        return CUDA(version=version, legacy=legacy)


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
    # Legacy variant string: `rocmxy`, new variant string: `rocm-x.y`.
    _LEGACY_VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"rocm(\d+)(\d+)")
    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"rocm-(\d+)\.(\d+)")

    version: Version
    legacy: bool = field(default=True, compare=False)

    @property
    def name(self) -> str:
        return "rocm"

    @property
    def variant_str(self) -> str:
        return _format_version("rocm", "rocm-", self.version, self.legacy)

    @staticmethod
    def parse(s: str, *, legacy: bool = True) -> "ROCm":
        version = _parse_version(ROCm._LEGACY_VARIANT_REGEX, ROCm._VARIANT_REGEX, s, legacy)
        if version is None:
            raise ValueError(f"Invalid ROCm variant string: {s!r}")
        return ROCm(version=version, legacy=legacy)


@strict
@dataclass(unsafe_hash=True)
class TPU:
    @property
    def name(self) -> str:
        return "tpu"

    @property
    def variant_str(self) -> str:
        return "tpu"

    @staticmethod
    def parse(s: str) -> "TPU":
        if s != "tpu":
            raise ValueError(f"Invalid TPU variant string: {s!r}")
        return TPU()


@dataclass(unsafe_hash=True)
class XPU:
    # Legacy variant string: `xpuxy`, new variant string: `xpu-x.y`.
    _LEGACY_VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"xpu(\d+)(\d+)")
    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"xpu-(\d+)\.(\d+)")

    version: Version
    legacy: bool = field(default=True, compare=False)

    @property
    def name(self) -> str:
        return "xpu"

    @property
    def variant_str(self) -> str:
        return _format_version("xpu", "xpu-", self.version, self.legacy)

    @staticmethod
    def parse(s: str, *, legacy: bool = True) -> "XPU":
        version = _parse_version(XPU._LEGACY_VARIANT_REGEX, XPU._VARIANT_REGEX, s, legacy)
        if version is None:
            raise ValueError(f"Invalid XPU variant string: {s!r}")
        return XPU(version=version, legacy=legacy)


_UNVERSIONED_BACKENDS: dict[str, type[CPU] | type[Metal] | type[Neuron] | type[TPU]] = {
    "cpu": CPU,
    "metal": Metal,
    "neuron": Neuron,
    "tpu": TPU,
}

_VERSIONED_BACKENDS: dict[str, type[CANN] | type[CUDA] | type[ROCm] | type[XPU]] = {
    "cann": CANN,
    "cuda": CUDA,
    "rocm": ROCm,
    "xpu": XPU,
}


def parse_backend(s: str, *, legacy: bool = True) -> Backend:
    """
    Parse a backend variant string into a Backend. In the legacy variant
    format, versions are part of a single token (e.g. `cu128`, `rocm61`,
    `cpu`). In the new variant format, the backend is written out in full and
    the version is separated by a dash (e.g. `cuda-12.8`, `rocm-6.1`, `cpu`).
    """
    if legacy:
        if s == "cpu":
            return CPU.parse(s)
        elif s == "metal":
            return Metal.parse(s)
        elif s == "neuron":
            return Neuron.parse(s)
        elif s == "tpu":
            return TPU.parse(s)
        elif s.startswith("cu"):
            return CUDA.parse(s)
        elif s.startswith("rocm"):
            return ROCm.parse(s)
        elif s.startswith("xpu"):
            return XPU.parse(s)
        elif s.startswith("cann"):
            return CANN.parse(s)
        raise ValueError(f"Unknown backend variant string: {s!r}")

    name = s.partition("-")[0]
    if name in _UNVERSIONED_BACKENDS:
        return _UNVERSIONED_BACKENDS[name].parse(s)
    elif name in _VERSIONED_BACKENDS:
        return _VERSIONED_BACKENDS[name].parse(s, legacy=False)
    raise ValueError(f"Unknown backend variant string: {s!r}")


def _backend() -> Backend:
    if has_torch:
        import torch

        acc = torch.accelerator.current_accelerator() if hasattr(torch, "accelerator") else None
        if acc is not None and acc.type == "tpu":
            return TPU()
        elif hasattr(torch, "neuron"):
            # Needs to be sorted before specific Torch builds, since Neuron
            # extension can be loaded into e.g. CUDA Torch builds.
            return Neuron()
        elif torch.version.cuda is not None:
            cuda_version = Version(torch.version.cuda)
            return CUDA(version=cuda_version)
        elif torch.version.hip is not None:
            # Note: the ROCm version does not always match the HIP version
            # (e.g. ROCm 10.0 uses HIP 7.15). So, we should use
            # `torch.version.rocm`. However, this attribute is only available
            # since Torch 2.10, so use `torch.version.hip` as a fallback.
            rocm = getattr(torch.version, "rocm", None) or torch.version.hip
            rocm_version = Version(rocm.split("-")[0])
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
        logger.warning("System has CUDA runtime library, but cannot get runtime version.")
        return None

    # cudaRuntimeGetVersion encodes the version as (major * 1000 + minor * 10).
    version_int = runtime_version.value
    major = version_int // 1000
    minor = (version_int % 1000) // 10

    return CUDA(version=Version(f"{major}.{minor}"))
