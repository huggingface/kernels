import platform
import re
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFolder
from packaging.version import Version, parse

from kernels.backends import (
    CANN,
    CUDA,
    XPU,
    Backend,
    ROCm,
    _select_backend,
    parse_backend,
)
from kernels.compat import has_torch, has_tvm_ffi

BUILD_VARIANT_REGEX = re.compile(
    r"^(torch\d+\d+|torch-(cpu|cuda|metal|neuron|rocm|xpu)|tvm-ffi\d+\d+)"
)

_TORCH_VARIANT_REGEX = re.compile(r"torch(\d+?)(\d+)")
_TVM_FFI_VARIANT_REGEX = re.compile(r"tvm-ffi(\d+?)(\d+)")


@dataclass(unsafe_hash=True)
class Torch:
    version: Version | None

    @property
    def variant_str(self) -> str:
        if self.version is None:
            return "torch"
        return f"torch{self.version.major}{self.version.minor}"

    @staticmethod
    def parse(s: str) -> "Torch":
        if s == "torch":
            return Torch(version=None)
        m = _TORCH_VARIANT_REGEX.fullmatch(s)
        if not m:
            raise ValueError(f"Invalid Torch variant string: {s!r}")
        return Torch(version=Version(f"{m.group(1)}.{m.group(2)}"))


@dataclass(unsafe_hash=True)
class TvmFfi:
    version: Version

    @property
    def variant_str(self) -> str:
        return f"tvm-ffi{self.version.major}{self.version.minor}"

    @staticmethod
    def parse(s: str) -> "TvmFfi":
        m = _TVM_FFI_VARIANT_REGEX.fullmatch(s)
        if not m:
            raise ValueError(f"Invalid TvmFfi variant string: {s!r}")
        return TvmFfi(version=Version(f"{m.group(1)}.{m.group(2)}"))


@dataclass
class Arch:
    backend: Backend
    platform: str
    os: str
    cxx11_abi: bool | None

    @property
    def variant_str(self) -> str:
        if self.cxx11_abi is None:
            return f"{self.backend.variant_str}-{self.platform}-{self.os}"
        else:
            return f"{'cxx11' if self.cxx11_abi else 'cxx98'}-{self.backend.variant_str}-{self.platform}-{self.os}"

    @staticmethod
    def parse(parts: list[str]) -> "Arch":
        # Handle Linux with cxx11 marker.
        if len(parts) == 4:
            cxx11_abi = parts[0] == "cxx11"
            parts = parts[1:]
        elif len(parts) == 3:
            cxx11_abi = None
        else:
            raise ValueError(f"Invalid arch variant parts: {parts!r}")

        backend = parse_backend(parts[0])
        platform = parts[1]
        os = parts[2]

        return Arch(backend=backend, platform=platform, os=os, cxx11_abi=cxx11_abi)


@dataclass
class Noarch:
    backend_name: str

    @property
    def variant_str(self) -> str:
        return self.backend_name

    @staticmethod
    def parse(s: str) -> "Noarch":
        return Noarch(backend_name=s)


@dataclass
class Variant:
    framework: Torch | TvmFfi
    arch: Arch | Noarch

    @property
    def variant_str(self) -> str:
        return f"{self.framework.variant_str}-{self.arch.variant_str}"

    @staticmethod
    def parse(variant_str: str) -> "Variant":
        parts = variant_str.split("-")

        arch: Arch | Noarch
        framework: Torch | TvmFfi

        if parts[0] == "torch":
            # noarch: e.g. "torch-cpu"
            framework = Torch.parse(parts[0])
            arch = Noarch.parse("-".join(parts[1:]))
        elif parts[0].startswith("torch"):
            framework = Torch.parse(parts[0])
            arch = Arch.parse(parts[1:])
        elif parts[0] == "tvm" and parts[1].startswith("ffi"):
            framework = TvmFfi.parse(f"tvm-{parts[1]}")
            arch = Arch.parse(parts[2:])
        else:
            raise ValueError(f"Unknown framework in variant string: {variant_str!r}")

        return Variant(framework=framework, arch=arch)


def get_variants(api: HfApi, *, repo_id: str, revision: str) -> list[Variant]:
    """Get all the build variants available from a kernel repository."""

    try:
        tree = api.list_repo_tree(repo_id, path_in_repo="build", revision=revision)
        variant_strs = {
            item.path.split("/")[-1] for item in tree if isinstance(item, RepoFolder)
        }
    except Exception:
        return []

    variants = []
    for variant_str in variant_strs:
        try:
            variants.append(Variant.parse(variant_str))
        except ValueError:
            pass
    return variants


def get_variants_local(repo_path: Path) -> list[Variant]:
    """Get all the build variants available in a local directory."""

    try:
        variant_strs = {entry.name for entry in repo_path.iterdir() if entry.is_dir()}
    except Exception:
        return []

    variants = []
    for variant_str in variant_strs:
        try:
            variants.append(Variant.parse(variant_str))
        except ValueError:
            pass
    return variants


def resolve_variant(
    variants: list[Variant], backend: str | None = None
) -> Variant | None:
    """Return the best matching variant for the current system."""
    selected_backend = _select_backend(backend)

    cpu = platform.machine()
    os = platform.system().lower()

    if os == "darwin":
        cpu = "aarch64" if cpu == "arm64" else cpu
    elif os == "windows":
        cpu = "x86_64" if cpu == "AMD64" else cpu

    torch_version = None
    torch_cxx11_abi = None
    if has_torch:
        import torch

        # Parse Torch version and strip patch/tags.
        torch_version = parse(torch.__version__)
        torch_version = Version(f"{torch_version.major}.{torch_version.minor}")

        torch_cxx11_abi = torch.compiled_with_cxx11_abi() if os == "linux" else None

    tvm_ffi_version = None
    if has_tvm_ffi:
        import tvm_ffi

        # Parse Torch version and strip patch/tags.
        tvm_ffi_version = parse(tvm_ffi.__version__)
        tvm_ffi_version = Version(f"{tvm_ffi_version.major}.{tvm_ffi_version.minor}")

    return _resolve_variant_for_system(
        variants=variants,
        selected_backend=selected_backend,
        cpu=cpu,
        os=os,
        torch_version=torch_version,
        torch_cxx11_abi=torch_cxx11_abi,
        tvm_ffi_version=tvm_ffi_version,
    )


def _resolve_variant_for_system(
    variants: list[Variant],
    selected_backend: Backend,
    cpu: str,
    os: str,
    torch_version: Version | None,
    torch_cxx11_abi: bool | None,
    tvm_ffi_version: Version | None,
) -> Variant | None:
    """Resolve the best matching variant given explicit system parameters."""
    applicable = _filter_variants(
        variants,
        selected_backend,
        cpu,
        os,
        torch_version,
        torch_cxx11_abi,
        tvm_ffi_version,
    )
    sorted_variants = _sort_variants(applicable)
    return sorted_variants[0] if sorted_variants else None


def _filter_variants(
    variants: list[Variant],
    selected_backend: Backend,
    cpu: str,
    os: str,
    torch_version: Version | None,
    torch_cxx11_abi: bool | None,
    tvm_ffi_version: Version | None,
) -> list[Variant]:
    """Return only the variants applicable to the current system."""
    result = []
    for v in variants:
        if isinstance(v.arch, Arch):
            # Skip non-matching CPU or OS.
            if v.arch.platform != cpu or v.arch.os != os:
                continue
            # If the variant is a Torch or tvm-ffi variant, check that it has the
            # correct version and ABI.
            if isinstance(v.framework, Torch):
                if v.framework.version != torch_version:
                    continue
                if v.arch.cxx11_abi != torch_cxx11_abi:
                    continue
            elif isinstance(v.framework, TvmFfi):
                if v.framework.version != tvm_ffi_version:
                    continue
            # Given a system CUDA version of x.y, only CUDA versions x.z,
            # where z <= y qualify. Otherwise, the backend + version (if present)
            # must match.
            if isinstance(selected_backend, CUDA) and isinstance(v.arch.backend, CUDA):
                if (
                    v.arch.backend.version.major != selected_backend.version.major
                    or v.arch.backend.version.minor > selected_backend.version.minor
                ):
                    continue
            elif v.arch.backend.variant_str != selected_backend.variant_str:
                continue
        else:
            assert isinstance(v.arch, Noarch)
            # Only noarch variants with a matching backend or "universal"
            # are applicable.
            noarch_backend_name = (
                "npu" if selected_backend.name == "cann" else selected_backend.name
            )
            if (
                v.arch.backend_name != noarch_backend_name
                and v.arch.backend_name != "universal"
            ):
                continue
        result.append(v)
    return result


def _sort_variants(
    variants: list[Variant],
) -> list[Variant]:
    """Sort variants in preference order:

    1. Torch arch kernels with with the highest compatible CUDA version.
    2. tvm-ffi arch kernels with with the highest compatible CUDA version.
    3. Torch noarch kernels.
    4. Old Torch universal kernels.
    """

    def sort_key(v: Variant) -> tuple:
        if isinstance(v.arch, Arch):
            framework_order = 0 if isinstance(v.framework, Torch) else 1
            if isinstance(v.arch.backend, (CUDA, ROCm, XPU, CANN)):
                # Order by backend version in reverse (higher is better).
                backend_order = -v.arch.backend.version.minor
            else:
                backend_order = 0
            return (framework_order, backend_order)
        else:
            assert isinstance(v.arch, Noarch)
            universal_order = 1 if v.arch.backend_name == "universal" else 0
            return (2, universal_order)

    return sorted(variants, key=sort_key)
