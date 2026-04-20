import itertools
import logging
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from huggingface_hub import HfApi
from huggingface_hub.dataclasses import strict
from huggingface_hub.hf_api import RepoFolder
from packaging.version import Version, parse

from kernels.backends import (
    CANN,
    CUDA,
    XPU,
    Backend,
    ROCm,
    _backend,
    _select_backend,
    parse_backend,
)
from kernels.compat import has_torch, has_tvm_ffi

BUILD_VARIANT_REGEX = re.compile(r"^(torch\d+\d+|torch-(cpu|cuda|metal|neuron|rocm|xpu)|tvm-ffi\d+\d+)")


@dataclass(unsafe_hash=True)
class Torch:
    """Versioned Torch framework (arch variants)."""

    # Match the following Torch version encoding:
    #
    # The first part is `torchxy` where x is the major version and y the minor
    # version. x can only consist of one digit, y of one or more digits.
    #
    # The optional second part is the ABI tag, which is `-cxx11` or `-cxx98`.
    # The ABI tag is used for historical reasons for Linux build variants. It
    # will be removed from builds in the future.
    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"torch(\d)(\d+)(?:-(cxx11|cxx98))?")

    version: Version
    cxx11_abi: bool | None

    @staticmethod
    def possible_variants() -> list["Torch"]:
        if has_torch:
            import torch

            torch_version = parse(torch.__version__)
            torch_version = Version(f"{torch_version.major}.{torch_version.minor}")

            os_ = platform.system().lower()
            if os_ == "linux":
                cxx11_abi = torch.compiled_with_cxx11_abi()
                return [
                    Torch(version=torch_version, cxx11_abi=cxx11_abi),
                    # We already accept build variants without an ABI tag, so
                    # that we can remove the tag from builds in the future.
                    Torch(version=torch_version, cxx11_abi=None),
                ]
            else:
                return [Torch(version=torch_version, cxx11_abi=None)]
        else:
            return []

    @property
    def variant_str(self) -> str:
        base = f"torch{self.version.major}{self.version.minor}"
        if self.cxx11_abi is None:
            return base
        return f"{base}-{'cxx11' if self.cxx11_abi else 'cxx98'}"

    @staticmethod
    def parse(s: str) -> "Torch":
        m = Torch._VARIANT_REGEX.fullmatch(s)
        if not m:
            raise ValueError(f"Invalid Torch variant string: {s!r}")
        version = Version(f"{m.group(1)}.{m.group(2)}")
        abi_str = m.group(3)
        if abi_str is None:
            cxx11_abi = None
        else:
            cxx11_abi = abi_str != "cxx98"
        return Torch(version=version, cxx11_abi=cxx11_abi)


@dataclass(unsafe_hash=True)
class TvmFfi:
    """Versioned tvm-ffi framework (arch variants)."""

    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"tvm-ffi(\d+?)(\d+)")

    version: Version

    @staticmethod
    def possible_variants() -> list["TvmFfi"]:
        if has_tvm_ffi:
            import tvm_ffi

            tvm_ffi_version = parse(tvm_ffi.__version__)
            tvm_ffi_version = Version(f"{tvm_ffi_version.major}.{tvm_ffi_version.minor}")
            return [TvmFfi(version=tvm_ffi_version)]
        else:
            return []

    @property
    def variant_str(self) -> str:
        return f"tvm-ffi{self.version.major}{self.version.minor}"

    @staticmethod
    def parse(s: str) -> "TvmFfi":
        m = TvmFfi._VARIANT_REGEX.fullmatch(s)
        if not m:
            raise ValueError(f"Invalid TvmFfi variant string: {s!r}")
        return TvmFfi(version=Version(f"{m.group(1)}.{m.group(2)}"))


@strict
@dataclass(unsafe_hash=True)
class TorchNoarch:
    """Versionless Torch framework (noarch variants)."""

    @staticmethod
    def possible_variants() -> list["TorchNoarch"]:
        if has_torch:
            return [TorchNoarch()]
        else:
            return []

    @property
    def variant_str(self) -> str:
        return "torch"


@strict
@dataclass(unsafe_hash=True)
class Arch:
    """Arch kernel information."""

    backend: Backend
    platform: str
    os: str

    @property
    def variant_str(self) -> str:
        return f"{self.backend.variant_str}-{self.platform}-{self.os}"

    @staticmethod
    def possible_variants() -> list["Arch"]:
        cpu = platform.machine()
        os = platform.system().lower()

        if os == "darwin":
            cpu = "aarch64" if cpu == "arm64" else cpu
        elif os == "windows":
            cpu = "x86_64" if cpu == "AMD64" else cpu

        backend = _backend()

        return [Arch(backend=backend, platform=cpu, os=os)]

    @staticmethod
    def parse(parts: list[str]) -> "Arch":
        if len(parts) != 3:
            raise ValueError(f"Invalid arch variant parts: {parts!r}")

        backend = parse_backend(parts[0])
        platform = parts[1]
        os = parts[2]

        return Arch(backend=backend, platform=platform, os=os)


@strict
@dataclass(unsafe_hash=True)
class Noarch:
    """Noarch kernel information."""

    backend_name: str

    @staticmethod
    def possible_variants() -> list["Noarch"]:
        backend = _backend()
        noarch_backend_name = "npu" if backend.name == "cann" else backend.name
        names = {noarch_backend_name, "universal"}
        return [Noarch(backend_name=name) for name in sorted(names)]

    @property
    def variant_str(self) -> str:
        return self.backend_name

    @staticmethod
    def parse(s: str) -> "Noarch":
        return Noarch(backend_name=s)


@strict
@dataclass(unsafe_hash=True)
class ArchVariant:
    """Arch kernel build variant."""

    framework: Torch | TvmFfi
    arch: Arch

    @staticmethod
    def possible_variants() -> list["ArchVariant"]:
        frameworks: list[Torch | TvmFfi] = Torch.possible_variants() + TvmFfi.possible_variants()
        archs = Arch.possible_variants()
        return [ArchVariant(framework=fw, arch=arch) for fw, arch in itertools.product(frameworks, archs)]

    @property
    def variant_str(self) -> str:
        return f"{self.framework.variant_str}-{self.arch.variant_str}"


@strict
@dataclass(unsafe_hash=True)
class NoarchVariant:
    """Noarch kernel build variant."""

    framework: TorchNoarch
    arch: Noarch

    @staticmethod
    def possible_variants() -> list["NoarchVariant"]:
        frameworks = TorchNoarch.possible_variants()
        archs = Noarch.possible_variants()
        return [NoarchVariant(framework=fw, arch=arch) for fw, arch in itertools.product(frameworks, archs)]

    @property
    def variant_str(self) -> str:
        return f"{self.framework.variant_str}-{self.arch.variant_str}"


Variant = ArchVariant | NoarchVariant


def system_variants() -> list[Variant]:
    """Return all possible build variants for the current system.

    Warning: this function should only be used internally (so don't export
             at the top-level) and for informational purposes, such as user
             feedback. When loading kernels, etc. rely what is on disk and
             use `parse_variant` + `resolve_variant`, since this uses our
             priority order, etc."""
    result: list[Variant] = ArchVariant.possible_variants() + NoarchVariant.possible_variants()
    return _sort_variants(result)


def parse_variant(variant_str: str) -> Variant:
    """Parse a variant string into an ArchVariant or NoarchVariant."""
    parts = variant_str.split("-")

    if parts[0] == "torch":
        # noarch: e.g. "torch-cpu"
        return NoarchVariant(framework=TorchNoarch(), arch=Noarch.parse("-".join(parts[1:])))
    elif parts[0].startswith("torch"):
        if len(parts) >= 2 and parts[1] in ("cxx11", "cxx98"):
            framework_str = f"{parts[0]}-{parts[1]}"
            arch_parts = parts[2:]
        else:
            framework_str = parts[0]
            arch_parts = parts[1:]
        return ArchVariant(framework=Torch.parse(framework_str), arch=Arch.parse(arch_parts))
    elif parts[0] == "tvm" and len(parts) >= 2 and parts[1].startswith("ffi"):
        return ArchVariant(framework=TvmFfi.parse(f"tvm-{parts[1]}"), arch=Arch.parse(parts[2:]))
    else:
        raise ValueError(f"Unknown framework in variant string: {variant_str!r}")


def get_variants(api: HfApi, *, repo_id: str, revision: str) -> list[Variant]:
    """Get all the build variants available from a kernel repository."""

    tree = api.list_repo_tree(repo_id, path_in_repo="build", repo_type="kernel", revision=revision)

    variant_strs = {item.path.split("/")[-1] for item in tree if isinstance(item, RepoFolder)}

    variants: list[Variant] = []
    for variant_str in variant_strs:
        try:
            variants.append(parse_variant(variant_str))
        except ValueError:
            logging.warning(
                f"Repository {repo_id} (revision: {revision}) contains invalid build variant variant: {variant_str!r}"
            )
    return variants


def get_variants_local(repo_path: Path) -> list[Variant]:
    """Get all the build variants available in a local directory."""

    try:
        variant_strs = {entry.name for entry in repo_path.iterdir() if entry.is_dir()}
    except Exception:
        return []

    variants: list[Variant] = []
    for variant_str in variant_strs:
        try:
            variants.append(parse_variant(variant_str))
        except ValueError:
            pass
    return variants


def resolve_variant(variants: list[Variant], backend: str | None = None) -> Variant | None:
    """Return the best matching variant for the current system."""
    resolved = resolve_variants(variants, backend)

    return resolved[0] if resolved else None


def resolve_variants(variants: list[Variant], backend: str | None = None) -> list[Variant]:
    """Return the matching variants for the current system, sorted
    by decreasing order of preference."""
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

        # Parse tvm-ffi version and strip patch/tags.
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
) -> list[Variant]:
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
    return _sort_variants(applicable)


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
    result: list[Variant] = []
    for v in variants:
        if isinstance(v, ArchVariant):
            # Skip non-matching CPU or OS.
            if v.arch.platform != cpu or v.arch.os != os:
                continue
            # If the variant is a Torch or tvm-ffi variant, check that it has the
            # correct version and ABI.
            if isinstance(v.framework, Torch):
                if v.framework.version != torch_version:
                    continue
                if v.framework.cxx11_abi is not None and v.framework.cxx11_abi != torch_cxx11_abi:
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
        elif isinstance(v, NoarchVariant):
            # Only noarch variants with a matching backend or "universal"
            # are applicable.
            noarch_backend_name = "npu" if selected_backend.name == "cann" else selected_backend.name
            if v.arch.backend_name != noarch_backend_name and v.arch.backend_name != "universal":
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
        if isinstance(v, ArchVariant):
            framework_order = 0 if isinstance(v.framework, Torch) else 1
            if isinstance(v.arch.backend, (CUDA, ROCm, XPU, CANN)):
                # Order by backend version in reverse (higher is better).
                backend_order = -v.arch.backend.version.minor
            else:
                backend_order = 0
            return (framework_order, backend_order)
        else:
            assert isinstance(v, NoarchVariant)
            universal_order = 1 if v.arch.backend_name == "universal" else 0
            return (2, universal_order)

    return sorted(variants, key=sort_key)
