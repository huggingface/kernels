import logging
import platform
import re
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, get_args

from huggingface_hub import HfApi, constants
from huggingface_hub.dataclasses import strict
from huggingface_hub.hf_api import RepoFolder
from packaging.version import Version, parse

from kernels._versions import resolve_revision_or_version
from kernels.backends import (
    CANN,
    CUDA,
    VERSIONED_BACKEND_NAMES,
    XPU,
    Backend,
    Metal,
    ROCm,
    _format_version,
    _parse_version,
    _select_backend,
    parse_backend,
)
from kernels.compat import has_torch, has_tvm_ffi

logger = logging.getLogger(__name__)

# Metal kernels are currently built with `-std=metal4.0`, which requires
# macOS 26 or later. Until the macOS/Metal version is encoded in the build
# variant, reject Metal arch variants on older systems, since they fail to
# load at runtime.
_METAL_MIN_MACOS_VERSION = Version("26.0")

# Python ABI of the compiled parts of a kernel:
#
# - `abi3`: stable ABI, for Python builds with a GIL.
# - `abi3t`: stable ABI for Python >= 3.15 free-threaded builds (PEP 803).
#            Also works with >= 3.15 GIL-enabled builds.
#
# Variants in the legacy variant string format do not have a Python ABI tag
# and are always `abi3`.
PythonAbi = Literal["abi3", "abi3t"]
_PYTHON_ABIS: tuple[str, ...] = get_args(PythonAbi)
_ABI3T_MIN_PYTHON_VERSION = Version("3.15")


@dataclass(unsafe_hash=True)
class Torch:
    """Versioned Torch framework (arch variants)."""

    # Match the following Torch version encodings:
    #
    # The first part is `torchxy` (legacy) or `torch-x.y` (new), where x is
    # the major version and y the minor version. In the legacy encoding, x
    # can only consist of one digit, y of one or more digits. In the new
    # encoding, both can consist of one or more digits.
    #
    # The optional second part is the ABI tag, which is `-cxx11` or `-cxx98`.
    # The ABI tag is used for historical reasons for Linux build variants. It
    # will be removed from builds in the future.
    _LEGACY_VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"torch(\d)(\d+)(?:-(cxx11|cxx98))?")
    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"torch-(\d+)\.(\d+)(?:-(cxx11|cxx98))?")

    version: Version
    cxx11_abi: bool | None
    # Whether to use the legacy variant string format (e.g. `torch210` rather
    # than `torch-2.10`). Does not take part in comparisons.
    legacy: bool = field(default=True, compare=False)

    @property
    def variant_str(self) -> str:
        base = _format_version("torch", "torch-", self.version, self.legacy)
        if self.cxx11_abi is None:
            return base
        return f"{base}-{'cxx11' if self.cxx11_abi else 'cxx98'}"

    @staticmethod
    def parse(s: str, *, legacy: bool = True) -> "Torch":
        m = (Torch._LEGACY_VARIANT_REGEX if legacy else Torch._VARIANT_REGEX).fullmatch(s)
        if not m:
            raise ValueError(f"Invalid Torch variant string: {s!r}")
        version = Version(f"{m.group(1)}.{m.group(2)}")
        abi_str = m.group(3)
        if abi_str is None:
            cxx11_abi = None
        else:
            cxx11_abi = abi_str != "cxx98"
        return Torch(version=version, cxx11_abi=cxx11_abi, legacy=legacy)


@dataclass(unsafe_hash=True)
class TorchStableAbi:
    """Stable ABI-versioned Torch framework (arch variants)."""

    # Match the following Torch version encodings:
    #
    # The first part is `torch-stable-abixy` (legacy) or
    # `torch-stable-abi-x.y` (new), where x is the major ABI version and y the
    # minor ABI version. In the legacy encoding, x can only consist of one
    # digit, y of one or more digits. In the new encoding, both can consist of
    # one or more digits.
    _LEGACY_VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"torch-stable-abi(\d)(\d+)")
    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"torch-stable-abi-(\d+)\.(\d+)")

    version: Version
    # Whether to use the legacy variant string format (e.g.
    # `torch-stable-abi211` rather than `torch-stable-abi-2.11`). Does not take
    # part in comparisons.
    legacy: bool = field(default=True, compare=False)

    @property
    def variant_str(self) -> str:
        return _format_version("torch-stable-abi", "torch-stable-abi-", self.version, self.legacy)

    @staticmethod
    def parse(s: str, *, legacy: bool = True) -> "TorchStableAbi":
        version = _parse_version(TorchStableAbi._LEGACY_VARIANT_REGEX, TorchStableAbi._VARIANT_REGEX, s, legacy)
        if version is None:
            raise ValueError(f"Invalid Torch stable ABI variant string: {s!r}")
        return TorchStableAbi(version=version, legacy=legacy)


@dataclass(unsafe_hash=True)
class TvmFfi:
    """Versioned tvm-ffi framework (arch variants)."""

    # Legacy variant string: `tvm-ffixy`, new variant string: `tvm-ffi-x.y`.
    _LEGACY_VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"tvm-ffi(\d+?)(\d+)")
    _VARIANT_REGEX: ClassVar[re.Pattern] = re.compile(r"tvm-ffi-(\d+)\.(\d+)")

    version: Version
    # Whether to use the legacy variant string format (e.g. `tvm-ffi01` rather
    # than `tvm-ffi-0.1`). Does not take part in comparisons.
    legacy: bool = field(default=True, compare=False)

    @property
    def variant_str(self) -> str:
        return _format_version("tvm-ffi", "tvm-ffi-", self.version, self.legacy)

    @staticmethod
    def parse(s: str, *, legacy: bool = True) -> "TvmFfi":
        version = _parse_version(TvmFfi._LEGACY_VARIANT_REGEX, TvmFfi._VARIANT_REGEX, s, legacy)
        if version is None:
            raise ValueError(f"Invalid TvmFfi variant string: {s!r}")
        return TvmFfi(version=version, legacy=legacy)


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
    python_abi: PythonAbi = "abi3"
    # Whether to use the legacy variant string format, which does not have a
    # Python ABI tag (the ABI is implicitly `abi3`). Does not take part in
    # comparisons.
    legacy: bool = field(default=True, compare=False)

    def __post_init__(self):
        if self.legacy and self.python_abi != "abi3":
            raise ValueError(f"The legacy variant string format only supports abi3, not {self.python_abi}")

    @property
    def variant_str(self) -> str:
        if self.legacy:
            return f"{self.backend.variant_str}-{self.platform}-{self.os}"
        return f"{self.backend.variant_str}-{self.python_abi}-{self.platform}-{self.os}"

    @staticmethod
    def parse(parts: list[str], *, legacy: bool = True) -> "Arch":
        """
        Parse arch parts: backend, Python ABI tag (new format only), platform
        and OS. In the legacy format, the backend (including its version) is a
        single part (`cu128`). In the new format, the backend name and the
        version are separate parts (`cuda`, `12.8`), unless the backend does
        not have a version, and the backend is followed by the Python ABI tag
        (`abi3` or `abi3t`).
        """
        backend_len = 2 if not legacy and parts and parts[0] in VERSIONED_BACKEND_NAMES else 1
        abi_len = 0 if legacy else 1
        if len(parts) != backend_len + abi_len + 2:
            raise ValueError(f"Invalid arch variant parts: {parts!r}")

        backend = parse_backend("-".join(parts[:backend_len]), legacy=legacy)
        python_abi = "abi3"
        if not legacy:
            python_abi = parts[backend_len]
            if python_abi not in _PYTHON_ABIS:
                raise ValueError(f"Invalid Python ABI tag {python_abi!r} in arch variant parts: {parts!r}")
        platform = parts[backend_len + abi_len]
        os = parts[backend_len + abi_len + 1]

        return Arch(backend=backend, platform=platform, os=os, python_abi=python_abi, legacy=legacy)  # type: ignore[arg-type]


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


@strict
@dataclass(unsafe_hash=True)
class Sdk:
    """Framework-less marker for SDK variants."""

    @property
    def variant_str(self) -> str:
        return "sdk"


@strict
@dataclass(unsafe_hash=True)
class SdkVariant:
    """SDK build variant.

    SDK variants have no framework. They only use the new variant string
    format. There are three forms:

    - `sdk`: framework and backend-agnostic (`arch` is `None`).
    - `sdk-<backend>`: SDK for a specific backend (`arch` is `Noarch`).
    - `sdk-<backend>[-<version>]-<abi>-<platform>-<os>`: SDK with compiled
      parts (`arch` is `Arch`), e.g. `sdk-cuda-12.8-abi3-x86_64-linux`.
    """

    framework: Sdk
    arch: Arch | Noarch | None

    @property
    def variant_str(self) -> str:
        if self.arch is None:
            return self.framework.variant_str
        return f"{self.framework.variant_str}-{self.arch.variant_str}"


Variant = ArchVariant | NoarchVariant | SdkVariant


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
    """Parse a variant string into an ArchVariant, NoarchVariant or SdkVariant.

    * Noarch variants have the shape `{framework}-{backend}`.
    * New-style arch variants have the shape
      `{framework}-{framework_version}-{backend}-{backend_version}-{python_abi}-{platform}-{os}`,
      unless the backend is versionless (e.g. CPU), then it has the shape
      `{framework}-{framework_version}-{backend}-{python_abi}-{platform}-{os}`.
      Versions in new-style variants are always noted as `{major}.{minor}`.
    * Legacy arch variants have no Python ABI tag (they are always abi3), no
      dash before versions and no dot inside versions. Also, `cu` is used as
      the backend name for CUDA.
    * SDKs follow noarch or new-style arch naming conventions, but use `sdk`
      in place of the framework and its version. Additionally, the variant
      name `sdk` can be used for SDKs that are backend-agnostic and do not
      have compiled parts.
    """
    parts = variant_str.split("-")

    if parts[0] == "sdk":
        return parse_sdk_variant(variant_str)
    if variant_str.startswith("torch-stable-abi"):
        # Legacy: `torch-stable-abi211-...`, new: `torch-stable-abi-2.11-...`.
        legacy = not (len(parts) > 3 and parts[2] == "abi")
        n = 3 if legacy else 4
        return ArchVariant(
            framework=TorchStableAbi.parse("-".join(parts[:n]), legacy=legacy),
            arch=Arch.parse(parts[n:], legacy=legacy),
        )
    if parts[0] == "torch":
        if len(parts) >= 2 and parts[1][:1].isdigit():
            # New format: `torch-2.10[-cxx11]-...`.
            n = 3 if len(parts) >= 3 and parts[2] in ("cxx11", "cxx98") else 2
            return ArchVariant(
                framework=Torch.parse("-".join(parts[:n]), legacy=False),
                arch=Arch.parse(parts[n:], legacy=False),
            )
        # noarch: e.g. "torch-cpu"
        return NoarchVariant(framework=TorchNoarch(), arch=Noarch.parse("-".join(parts[1:])))
    elif parts[0].startswith("torch"):
        # Legacy format: `torch210[-cxx11]-...`.
        if len(parts) >= 2 and parts[1] in ("cxx11", "cxx98"):
            framework_str = f"{parts[0]}-{parts[1]}"
            arch_parts = parts[2:]
        else:
            framework_str = parts[0]
            arch_parts = parts[1:]
        return ArchVariant(framework=Torch.parse(framework_str), arch=Arch.parse(arch_parts))
    elif parts[0] == "tvm" and len(parts) >= 2 and parts[1].startswith("ffi"):
        # Legacy: `tvm-ffi01-...`, new: `tvm-ffi-0.1-...`.
        legacy = not (len(parts) > 2 and parts[1] == "ffi")
        n = 2 if legacy else 3
        return ArchVariant(
            framework=TvmFfi.parse("-".join(parts[:n]), legacy=legacy),
            arch=Arch.parse(parts[n:], legacy=legacy),
        )
    else:
        raise ValueError(f"Unknown framework in variant string: {variant_str!r}")


def parse_sdk_variant(variant_str: str) -> SdkVariant:
    """Parse a variant string into an SdkVariant."""
    parts = variant_str.split("-")

    if parts[0] != "sdk":
        raise ValueError(f"Not an SDK variant string: {variant_str!r}")

    arch: Arch | Noarch | None
    if len(parts) == 1:
        arch = None
    elif len(parts) == 2:
        arch = Noarch.parse(parts[1])
    else:
        arch = Arch.parse(parts[1:], legacy=False)

    return SdkVariant(framework=Sdk(), arch=arch)


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

    macos_version = None
    if os == "darwin":
        cpu = "aarch64" if cpu == "arm64" else cpu
        mac_release = platform.mac_ver()[0]
        if mac_release:
            macos_version = parse(mac_release)
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
        macos_version=macos_version,
        python_version=Version(f"{sys.version_info.major}.{sys.version_info.minor}"),
        free_threaded=_is_free_threaded_build(),
    )


def _resolve_variant_for_system(
    variants: list[Variant],
    selected_backend: Backend,
    cpu: str,
    os: str,
    torch_version: Version | None,
    torch_cxx11_abi: bool | None,
    tvm_ffi_version: Version | None,
    macos_version: Version | None = None,
    python_version: Version = _ABI3T_MIN_PYTHON_VERSION,
    free_threaded: bool = False,
) -> tuple[list[Variant], list[Decision]]:
    """Resolve the best matching variant given explicit system parameters.

    The Python version and free-threaded state default to a GIL-enabled
    Python 3.15, which supports both abi3 and abi3t variants.

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
        macos_version,
        python_version,
        free_threaded,
    )
    trace = _sort_variants(trace)

    applicable = [decision.variant for decision in trace if isinstance(decision, VariantAccepted)]
    return applicable, trace


def _is_free_threaded_build() -> bool:
    """Check if the Python interpreter is a free-threaded build."""
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def _python_abi_mismatch(python_abi: str, python_version: Version, free_threaded: bool) -> str | None:
    """Return the reason if the Python ABI is not supported by the interpreter."""
    if python_abi == "abi3":
        # Free-threaded builds do not load abi3 extensions.
        if free_threaded:
            return "abi3 is not supported on free-threaded Python, abi3t is required"
    elif python_version < _ABI3T_MIN_PYTHON_VERSION:
        return f"abi3t requires Python {_ABI3T_MIN_PYTHON_VERSION} or later, system Python version is {python_version}"
    return None


def _platform_mismatch(arch: Arch, cpu: str, os: str) -> str | None:
    """Return the reason if the arch's CPU or OS does not match the system."""
    if arch.platform != cpu:
        return f"CPU ({arch.platform}) does not match system CPU ({cpu})"
    if arch.os != os:
        return f"OS ({arch.os}) does not match system OS ({os})"
    return None


def _backend_mismatch(backend: Backend, selected_backend: Backend, macos_version: Version | None) -> str | None:
    """Return the reason if the variant's backend is not compatible with the
    selected backend."""
    # Given a system CUDA version of x.y, only CUDA versions x.z,
    # where z <= y qualify. Otherwise, the backend + version (if present)
    # must match.
    if isinstance(selected_backend, CUDA) and isinstance(backend, CUDA):
        if (
            backend.version.major != selected_backend.version.major
            or backend.version.minor > selected_backend.version.minor
        ):
            return f"CUDA version ({backend.version}) is not compatible with system CUDA version ({selected_backend.version})"
    # Same for ROCm: given a system ROCm version of x.y, only ROCm
    # versions x.z, where z <= y qualify.
    elif isinstance(selected_backend, ROCm) and isinstance(backend, ROCm):
        if (
            backend.version.major != selected_backend.version.major
            or backend.version.minor > selected_backend.version.minor
        ):
            return f"ROCm version ({backend.version}) is not compatible with system ROCm version ({selected_backend.version})"
    elif backend != selected_backend:
        return f"backend ({backend.variant_str}) does not match selected backend ({selected_backend.variant_str})"
    elif isinstance(backend, Metal) and (macos_version is None or macos_version < _METAL_MIN_MACOS_VERSION):
        return (
            f"Metal kernels require macOS {_METAL_MIN_MACOS_VERSION} or later, system macOS version is {macos_version}"
        )
    return None


def _check_variants(
    variants: list[Variant],
    selected_backend: Backend,
    cpu: str,
    os: str,
    torch_version: Version | None,
    torch_cxx11_abi: bool | None,
    tvm_ffi_version: Version | None,
    macos_version: Version | None = None,
    python_version: Version = _ABI3T_MIN_PYTHON_VERSION,
    free_threaded: bool = False,
) -> list[Decision]:
    """Return only the variants applicable to the current system."""
    result: list[Decision] = []
    for v in variants:
        if isinstance(v, ArchVariant):
            # Skip variants with a Python ABI that the interpreter cannot load.
            reason = _python_abi_mismatch(v.arch.python_abi, python_version, free_threaded)
            if reason is not None:
                result.append(VariantRejected(variant=v, reason=reason))
                continue

            # Skip non-matching CPU or OS.
            reason = _platform_mismatch(v.arch, cpu, os)
            if reason is not None:
                result.append(VariantRejected(variant=v, reason=reason))
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
            reason = _backend_mismatch(v.arch.backend, selected_backend, macos_version)
            if reason is not None:
                result.append(VariantRejected(variant=v, reason=reason))
                continue
        elif isinstance(v, NoarchVariant):
            # Only noarch variants with a matching backend or "universal"
            # are applicable.
            noarch_backend_name = selected_backend.name
            if v.arch.backend_name != noarch_backend_name and v.arch.backend_name != "universal":
                result.append(
                    VariantRejected(
                        variant=v,
                        reason=f"backend ({v.arch.backend_name}) does not match system backend ({noarch_backend_name}) and is not universal",
                    )
                )
                continue
        elif isinstance(v, SdkVariant):
            if v.arch is None:
                # Framework and backend-agnostic.
                pass
            elif isinstance(v.arch, Noarch):
                if v.arch.backend_name != selected_backend.name:
                    result.append(
                        VariantRejected(
                            variant=v,
                            reason=f"backend ({v.arch.backend_name}) does not match system backend ({selected_backend.name})",
                        )
                    )
                    continue
            else:
                reason = (
                    _python_abi_mismatch(v.arch.python_abi, python_version, free_threaded)
                    or _platform_mismatch(v.arch, cpu, os)
                    or _backend_mismatch(v.arch.backend, selected_backend, macos_version)
                )
                if reason is not None:
                    result.append(VariantRejected(variant=v, reason=reason))
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
    4. SDK arch kernels with the highest compatible CUDA version.
    5. Torch noarch kernels.
    6. Old Torch universal kernels.
    7. SDK kernels for a specific backend.
    8. SDK kernels that are framework and backend-agnostic.

    Arch kernels that are otherwise equal are ordered by Python ABI: abi3
    before abi3t, since GIL-enabled builds prefer abi3 (abi3 variants are never
    accepted on free-threaded builds).
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
            python_abi_order = _PYTHON_ABIS.index(v.arch.python_abi)
            return (decision_order, framework_order, *abi_version_order, backend_order, python_abi_order)
        elif isinstance(v, SdkVariant):
            if isinstance(v.arch, Arch):
                backend_order = (
                    -v.arch.backend.version.minor if isinstance(v.arch.backend, (CUDA, ROCm, XPU, CANN)) else 0
                )
                python_abi_order = _PYTHON_ABIS.index(v.arch.python_abi)
                return (decision_order, 3, 0, 0, backend_order, python_abi_order)
            elif isinstance(v.arch, Noarch):
                return (decision_order, 5, 0, 0, 0, 0)
            return (decision_order, 6, 0, 0, 0, 0)
        else:
            assert isinstance(v, NoarchVariant)
            universal_order = 1 if v.arch.backend_name == "universal" else 0
            return (decision_order, 4, 0, 0, universal_order, 0)

    return sorted(variants, key=sort_key)


def get_kernel_variants(
    repo_id: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
) -> list[Decision]:
    """
    Resolve all build variants of a kernel against the current environment.

    The decisions are sorted with compatible variants first, the most preferred
    variant leading.

    Args:
        repo_id (`str`):
            The Hub repository containing the kernel.
        revision (`str`, *optional*):
            The specific revision (branch, tag, or commit) to inspect. Cannot be used together with `version`.
        version (`int`, *optional*):
            The kernel version to inspect. Cannot be used together with `revision`.
            Either `version` or `revision` must be specified.
        backend (`str`, *optional*):
            The backend to resolve variants for. Can only be `cpu` or the backend that Torch is compiled for.
            The backend will be detected automatically if not provided.

    Returns:
        `list[Decision]`: One `VariantAccepted` or `VariantRejected` per build variant
            in the repository, compatible variants first.

    Example:
        ```python
        from kernels import get_kernel_variants, VariantAccepted

        for decision in get_kernel_variants("kernels-community/activation", version=1):
            name = decision.variant.variant_str
            if isinstance(decision, VariantAccepted):
                print(f"{name}: compatible")
            else:
                print(f"{name}: rejected ({decision.reason})")

        ```
    """
    from kernels.hf_hub import _get_hf_api

    commit = resolve_revision_or_version(
        repo_id,
        revision=revision,
        version=version,
        local_files_only=constants.HF_HUB_OFFLINE,
    )

    api = _get_hf_api()
    variants = get_variants(api, repo_id=repo_id, revision=str(commit))
    _, trace = resolve_variants(variants, backend)
    return trace


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
