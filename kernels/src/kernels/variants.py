import logging
import platform
import re
import sys
import sysconfig
import warnings
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
    _select_backend,
    parse_backend,
)
from kernels.compat import has_torch, has_tvm_ffi


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
class TorchStableAbi:
    """Stable ABI-versioned Torch framework (arch variants)."""

    # Match the following Torch version encoding:
    #
    # The first part is `torch-stable-abixy` where x is the major ABI version
    # and y the minor ABI version. x can only consist of one digit, y of one
    # or more digits.
    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"torch-stable-abi(\d)(\d+)")

    version: Version

    @property
    def variant_str(self) -> str:
        return f"torch-stable-abi{self.version.major}{self.version.minor}"

    @staticmethod
    def parse(s: str) -> "TorchStableAbi":
        m = TorchStableAbi._VARIANT_REGEX.fullmatch(s)
        if not m:
            raise ValueError(f"Invalid Torch stable ABI variant string: {s!r}")
        version = Version(f"{m.group(1)}.{m.group(2)}")
        return TorchStableAbi(version=version)


@dataclass(unsafe_hash=True)
class TvmFfi:
    """Versioned tvm-ffi framework (arch variants)."""

    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"tvm-ffi(\d+?)(\d+)")

    version: Version

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

    framework: Torch | TorchStableAbi | TvmFfi
    arch: Arch

    @property
    def variant_str(self) -> str:
        return f"{self.framework.variant_str}-{self.arch.variant_str}"


@strict
@dataclass(unsafe_hash=True)
class NoarchVariant:
    """Noarch kernel build variant."""

    framework: TorchNoarch
    arch: Noarch

    @property
    def variant_str(self) -> str:
        return f"{self.framework.variant_str}-{self.arch.variant_str}"


Variant = ArchVariant | NoarchVariant


@dataclass(unsafe_hash=True)
class VariantAccepted:
    """Variant that is compatible with the current system."""

    variant: Variant


@dataclass(unsafe_hash=True)
class VariantRejected:
    """Variant that is incompatible with the current system."""

    variant: Variant
    reason: str


Decision = VariantAccepted | VariantRejected


def parse_variant(variant_str: str) -> Variant:
    """Parse a variant string into an ArchVariant or NoarchVariant."""
    parts = variant_str.split("-")

    if variant_str.startswith("torch-stable-abi"):
        return ArchVariant(
            framework=TorchStableAbi.parse("-".join(parts[0:3])),
            arch=Arch.parse(parts[3:]),
        )
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


def resolve_variant(variants: list[Variant], backend: str | None = None) -> tuple[Variant | None, list[Decision]]:
    """Return the best matching variant for the current system and
    a trace with the acceptance/rejection decision for each variant."""
    resolved, trace = resolve_variants(variants, backend)

    return resolved[0] if resolved else None, trace


def resolve_variants(variants: list[Variant], backend: str | None = None) -> tuple[list[Variant], list[Decision]]:
    """Return the matching variants for the current system, sorted
    by decreasing order of preference, and a trace of the
    acceptance/rejection decision for each variant."""
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
) -> tuple[list[Variant], list[Decision]]:
    """Resolve the best matching variant given explicit system parameters.

    Returns the preference-sorted list of accepted variants and a trace of
    the acceptance/rejection decision for each variant."""
    trace = _check_variants(
        variants,
        selected_backend,
        cpu,
        os,
        torch_version,
        torch_cxx11_abi,
        tvm_ffi_version,
    )
    trace = _sort_variants(trace)

    applicable = [decision.variant for decision in trace if isinstance(decision, VariantAccepted)]
    return applicable, trace


def _is_unsupported_free_threaded_build() -> bool:
    """Check if the Python interpreter is a free-threaded build that does not
    support ABI3."""
    return sys.version_info < (3, 15) and bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def _check_variants(
    variants: list[Variant],
    selected_backend: Backend,
    cpu: str,
    os: str,
    torch_version: Version | None,
    torch_cxx11_abi: bool | None,
    tvm_ffi_version: Version | None,
) -> list[Decision]:
    """Return only the variants applicable to the current system."""
    is_unsupported_free_threaded = _is_unsupported_free_threaded_build()
    # Prefilter all arch kernels on free-threaded Python pre-3.15, since
    # they do not support the stable ABI.
    if is_unsupported_free_threaded:
        warnings.warn(
            "Arch kernels use the stable ABI, which is not supported on free-threaded "
            "Python before version 3.15. Arch kernels will not be used. Consider using "
            "a non-free-threaded interpreter, or upgrade to Python 3.15+.",
            UserWarning,
            stacklevel=2,
        )
        variants = [v for v in variants if not isinstance(v, ArchVariant)]

    result: list[Decision] = []
    for v in variants:
        if isinstance(v, ArchVariant):
            if is_unsupported_free_threaded:
                result.append(
                    VariantRejected(
                        variant=v,
                        reason="arch kernel not supported on free-threaded Python <3.15",
                    )
                )
                continue

            # Skip non-matching CPU or OS.
            if v.arch.platform != cpu:
                result.append(
                    VariantRejected(
                        variant=v,
                        reason=f"CPU ({v.arch.platform}) does not match system CPU ({cpu})",
                    )
                )
                continue
            if v.arch.os != os:
                result.append(
                    VariantRejected(
                        variant=v,
                        reason=f"OS ({v.arch.os}) does not match system OS ({os})",
                    )
                )
                continue
            # If the variant is a Torch or tvm-ffi variant, check that it has the
            # correct version and ABI.
            if isinstance(v.framework, Torch):
                if v.framework.version != torch_version:
                    result.append(
                        VariantRejected(
                            variant=v,
                            reason=f"Torch version ({v.framework.version}) does not match environment Torch version ({torch_version})",
                        )
                    )
                    continue
                if v.framework.cxx11_abi is not None and v.framework.cxx11_abi != torch_cxx11_abi:
                    result.append(
                        VariantRejected(
                            variant=v,
                            reason=f"Torch CXX11 ABI ({v.framework.cxx11_abi}) does not match environment Torch CXX11 ABI ({torch_cxx11_abi})",
                        )
                    )
                    continue
            elif isinstance(v.framework, TorchStableAbi):
                if torch_version is None or v.framework.version > torch_version:
                    result.append(
                        VariantRejected(
                            variant=v,
                            reason=f"Torch stable ABI version ({v.framework.version}) is too new for environment Torch version ({torch_version})",
                        )
                    )
                    continue
            elif isinstance(v.framework, TvmFfi):
                if v.framework.version != tvm_ffi_version:
                    result.append(
                        VariantRejected(
                            variant=v,
                            reason=f"tvm-ffi version ({v.framework.version}) does not match environment tvm-ffi version ({tvm_ffi_version})",
                        )
                    )
                    continue
            # Given a system CUDA version of x.y, only CUDA versions x.z,
            # where z <= y qualify. Otherwise, the backend + version (if present)
            # must match.
            if isinstance(selected_backend, CUDA) and isinstance(v.arch.backend, CUDA):
                if (
                    v.arch.backend.version.major != selected_backend.version.major
                    or v.arch.backend.version.minor > selected_backend.version.minor
                ):
                    result.append(
                        VariantRejected(
                            variant=v,
                            reason=f"CUDA version ({v.arch.backend.version}) is not compatible with system CUDA version ({selected_backend.version})",
                        )
                    )
                    continue
            elif v.arch.backend.variant_str != selected_backend.variant_str:
                result.append(
                    VariantRejected(
                        variant=v,
                        reason=f"backend ({v.arch.backend.variant_str}) does not match selected backend ({selected_backend.variant_str})",
                    )
                )
                continue
        elif isinstance(v, NoarchVariant):
            # Only noarch variants with a matching backend or "universal"
            # are applicable.
            noarch_backend_name = "npu" if selected_backend.name == "cann" else selected_backend.name
            if v.arch.backend_name != noarch_backend_name and v.arch.backend_name != "universal":
                result.append(
                    VariantRejected(
                        variant=v,
                        reason=f"backend ({v.arch.backend_name}) does not match system backend ({noarch_backend_name}) and is not universal",
                    )
                )
                continue
        result.append(VariantAccepted(variant=v))
    return result


def _sort_variants(
    variants: list[Decision],
) -> list[Decision]:
    """Sort the decision trace in preference order:

    1. AcceptedVariant before RejectedVariant.
    2. Torch stable ABI arch kernels, with highest compatible version first,
       then highest compatible CUDA version.
    2. Torch arch kernels (tagless before C++ ABI-tagged) with the highest compatible CUDA version.
    3. tvm-ffi arch kernels with with the highest compatible CUDA version.
    4. Torch noarch kernels.
    5. Old Torch universal kernels.
    """

    def sort_key(vs: Decision) -> tuple[int, ...]:
        # Returns a tuple of ints used for comparison.
        decision_order = 0 if isinstance(vs, VariantAccepted) else 1
        v = vs.variant
        if isinstance(v, ArchVariant):
            if isinstance(v.framework, TorchStableAbi):
                framework_order = 0
                # Prefer newer stable ABI versions.
                abi_version_order = (
                    -v.framework.version.major,
                    -v.framework.version.minor,
                )
            elif isinstance(v.framework, Torch):
                framework_order = 1
                # Prefer tagless (cxx11_abi is None) over ABI-tagged.
                abi_version_order = (0, 1 if v.framework.cxx11_abi is not None else 0)
            else:
                framework_order = 2
                abi_version_order = (0, 0)
            if isinstance(v.arch.backend, (CUDA, ROCm, XPU, CANN)):
                # Order by backend version in reverse (higher is better).
                backend_order = -v.arch.backend.version.minor
            else:
                backend_order = 0
            return (decision_order, framework_order, *abi_version_order, backend_order)
        else:
            assert isinstance(v, NoarchVariant)
            universal_order = 1 if v.arch.backend_name == "universal" else 0
            return (decision_order, 2, 0, 0, universal_order)

    return sorted(variants, key=sort_key)


def variants_trace_str(trace: list[Decision]) -> str:
    # Ensure that the list is sorted.
    sorted = _sort_variants(trace)
    best = sorted[0].variant if len(sorted) and isinstance(sorted[0], VariantAccepted) else None
    return "\n".join(
        [
            (
                f"{variant_trace.variant.variant_str} {'compatible, preferred' if variant_trace.variant == best else 'compatible'} ✅"
                if isinstance(variant_trace, VariantAccepted)
                else f"{variant_trace.variant.variant_str}: {variant_trace.reason}"
            )
            for variant_trace in trace
        ]
    )
