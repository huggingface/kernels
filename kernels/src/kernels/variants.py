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
    XPU,
    Backend,
    Metal,
    ROCm,
    _format_version,
    _parse_version,
    _select_backend,
    parse_backend,
)
from kernels.compat import has_mlx, has_torch, has_tvm_ffi

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

# In variant strings, the Python version and ABI are preceded by the marker
# `cpython`, e.g. `cpython-3.12-abi3`.
_PYTHON_MARKER = "cpython"
_PYTHON_VERSION_REGEX = re.compile(r"(\d+)\.(\d+)")

# Minimum Python version for abi3t.
_ABI3T_MIN_PYTHON_VERSION = Version("3.15")

# ABI version for legacy build variants.
_LEGACY_PYTHON_VERSION = Version("3.9")


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
class MlxNoarch:
    """Versionless MLX framework (noarch variants)."""

    @property
    def variant_str(self) -> str:
        return "mlx"


@strict
@dataclass(unsafe_hash=True)
class Python:
    """The Python version and ABI that the compiled parts of a kernel or
    or SDK are build for."""

    # The minimum Python version that is supported by the compiled parts.
    version: Version = _LEGACY_PYTHON_VERSION
    abi: PythonAbi = "abi3"
    # Whether to use the legacy variant string format, which does not have a
    # Python version and ABI (they are implicitly `cpython-3.9-abi3`).
    legacy: bool = field(default=True, compare=False)

    def __post_init__(self):
        if self.legacy and (self.abi != "abi3" or self.version != _LEGACY_PYTHON_VERSION):
            raise ValueError(
                f"The legacy variant string format only supports Python {_LEGACY_PYTHON_VERSION} and abi3, "
                f"not Python {self.version} and {self.abi}"
            )
        if self.abi == "abi3t" and self.version < _ABI3T_MIN_PYTHON_VERSION:
            raise ValueError(f"abi3t requires Python {_ABI3T_MIN_PYTHON_VERSION} or later, not {self.version}")

    @property
    def variant_str(self) -> str:
        """The variant string, which is empty in the legacy format."""
        if self.legacy:
            return ""
        return f"{_PYTHON_MARKER}-{self.version.major}.{self.version.minor}-{self.abi}"

    @staticmethod
    def parse(parts: list[str]) -> "Python":
        """
        Parse the Python version and ABI of the new variant string format:

        * The marker `cpython`.
        * The Python version (e.g. `3.12`).
        * The ABI tag (`abi3` or `abi3t`).
        """
        if len(parts) != 3:
            raise ValueError(
                f"Invalid Python variant parts, expected `{_PYTHON_MARKER}-[version]-[abi]`, got: {parts!r}"
            )

        marker, version, abi = parts
        if marker != _PYTHON_MARKER:
            raise ValueError(f"Invalid Python variant parts, expected {_PYTHON_MARKER!r}, got: {marker!r}")
        m = _PYTHON_VERSION_REGEX.fullmatch(version)
        if m is None:
            raise ValueError(f"Invalid Python version, expected `[major].[minor]`, got: {version!r}")
        if abi not in _PYTHON_ABIS:
            raise ValueError(f"Invalid ABI tag {abi!r} in Python variant parts, expected one of: {_PYTHON_ABIS!r}")

        return Python(version=Version(f"{m.group(1)}.{m.group(2)}"), abi=abi, legacy=False)  # type: ignore[arg-type]


@strict
@dataclass(unsafe_hash=True)
class System:
    """The system that variants with compiled parts are built for: the platform
    (CPU architecture) and the OS."""

    platform: str
    os: str

    @property
    def variant_str(self) -> str:
        return f"{self.platform}-{self.os}"

    @staticmethod
    def parse(parts: list[str]) -> "System":
        """
        Parse system parts:

        * The platform (e.g. `x86_64`)
        * The OS (e.g. `linux`).
        """
        if len(parts) != 2:
            raise ValueError(f"Invalid system variant parts, expected `[platform]-[os]`, got: {parts!r}")

        return System(platform=parts[0], os=parts[1])


@strict
@dataclass(unsafe_hash=True)
class Noarch:
    """Backend of noarch variants. Noarch variants only specify the name of
    the backend, since they do not have compiled parts."""

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
    backend: Backend
    python: Python
    system: System

    @property
    def variant_str(self) -> str:
        parts = [self.framework.variant_str, self.backend.variant_str]
        # The Python version and ABI are implicit in the legacy format.
        if not self.python.legacy:
            parts.append(self.python.variant_str)
        parts.append(self.system.variant_str)
        return "-".join(parts)


@strict
@dataclass(unsafe_hash=True)
class NoarchVariant:
    """Noarch kernel build variant."""

    framework: MlxNoarch | TorchNoarch
    backend: Noarch

    @property
    def variant_str(self) -> str:
        return f"{self.framework.variant_str}-{self.backend.variant_str}"


@strict
@dataclass(unsafe_hash=True)
class SdkVariant:
    """SDK build variant.

    SDK variants have no framework. They only use the new variant string
    format. There are four forms:

    - `sdk`: framework and backend-agnostic, no dynamic libraries.
    - `sdk-<backend>`: SDK for a specific backend, no dynamic libraries.
    - `sdk-cpython-<python_version>-<abi>-<platform>-<os>`: SDK with dynamic
      libraries that works with any backend, e.g.
      `sdk-cpython-3.12-abi3-x86_64-linux`.
    - `sdk-<backend>[-<version>]-cpython-<python_version>-<abi>-<platform>-<os>`:
      SDK with dynamic libraries for a specific backend, e.g.
      `sdk-cuda-12.8-cpython-3.12-abi3-x86_64-linux`.
    """

    backend: Backend | Noarch | None
    python: Python | None
    system: System | None

    def __post_init__(self):
        if (self.python is None) != (self.system is None):
            raise ValueError("SDK variants must define the Python version and ABI together with the system")
        # Only backends with dynamic libraries use versioned backends.
        compiled = self.system is not None
        if isinstance(self.backend, Noarch) and compiled:
            raise ValueError("SDK variants that define a system must also have a versioned backend")
        if self.backend is not None and not isinstance(self.backend, Noarch) and not compiled:
            raise ValueError("SDK variants with a versioned backend must also define a system")

    @property
    def variant_str(self) -> str:
        parts = ["sdk"]
        if self.backend is not None:
            parts.append(self.backend.variant_str)
        if self.python is not None and not self.python.legacy:
            parts.append(self.python.variant_str)
        if self.system is not None:
            parts.append(self.system.variant_str)
        return "-".join(parts)


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


def _split_system(parts: list[str]) -> tuple[list[str], System]:
    """Split the system (platform and OS) off the end of the variant parts.
    Returns the remaining parts and the system.

    For example this function applied to
    `["cuda", 13.0", "cpython", "3.9", "abi3", "x86_64", "linux"]`
    returns `(["cuda", "13.0", "cpython", "3.9", "abi3"], System(...))`.
    """
    if len(parts) < 2:
        raise ValueError(f"Invalid variant parts: {parts!r}")
    return parts[:-2], System.parse(parts[-2:])


def _split_python(parts: list[str], *, legacy: bool) -> tuple[list[str], Python | None]:
    """Split the Python version and ABI off the end of the variant parts.
    Returns the remaining parts and the Python version and ABI.

    In the new format, the Python version and ABI start at the `cpython`
    marker. For example this function applied to
    `["cuda", 13.0", "cpython", "3.9", "abi3"]` returns
    `(["cuda", "13.0"], Python(...))`.

    The result is `None` if the variant string does not have a Python ABI
    version. In the legacy format, the Python version and ABI are implicit.
    """
    if legacy:
        return parts, Python()

    if _PYTHON_MARKER not in parts:
        return parts, None
    i = parts.index(_PYTHON_MARKER)
    return parts[:i], Python.parse(parts[i:])


def parse_variant(variant_str: str) -> Variant:
    """Parse a variant string into an ArchVariant, NoarchVariant or SdkVariant.

    * Noarch variants have the shape `{framework}-{backend}`.
    * New-style arch variants have the shape
      `{framework}-{framework_version}-{backend}-{backend_version}-cpython-{python_version}-{python_abi}-{platform}-{os}`,
      unless the backend is versionless (e.g. CPU), then it has the shape
      `{framework}-{framework_version}-{backend}-cpython-{python_version}-{python_abi}-{platform}-{os}`.
      Versions in new-style variants are always noted as `{major}.{minor}`.
      `cpython-{python_version}` is the minimum Python version (e.g.
      `cpython-3.12` for Python 3.12 or later).
    * Legacy arch variants have no Python version and ABI (they are always
      `cpython-3.9-abi3`), no dash before versions and no dot inside versions.
      Also, `cu` is used as the backend name for CUDA.
    * SDKs follow noarch or new-style arch naming conventions, but use `sdk`
      in place of the framework and its version. Additionally, the variant
      name `sdk` can be used for SDKs that are backend-agnostic and do not
      have compiled parts.
    """
    parts = variant_str.split("-")

    # Each branch only parses the framework, determining the number of parts
    # that the framework takes and whether the legacy format is used.
    framework: Torch | TorchStableAbi | TvmFfi
    if parts[0] == "sdk":
        return parse_sdk_variant(variant_str)
    elif variant_str.startswith("torch-stable-abi"):
        # Legacy: `torch-stable-abi211-...`, new: `torch-stable-abi-2.11-...`.
        legacy = not (len(parts) > 3 and parts[2] == "abi")
        n = 3 if legacy else 4
        framework = TorchStableAbi.parse("-".join(parts[:n]), legacy=legacy)
    elif parts[0] == "torch" and len(parts) >= 2 and parts[1][:1].isdigit():
        # New format: `torch-2.10[-cxx11]-...`.
        legacy = False
        n = 3 if len(parts) >= 3 and parts[2] in ("cxx11", "cxx98") else 2
        framework = Torch.parse("-".join(parts[:n]), legacy=legacy)
    elif parts[0] == "torch":
        # noarch: e.g. "torch-cpu"
        return NoarchVariant(framework=TorchNoarch(), backend=Noarch.parse("-".join(parts[1:])))
    elif parts[0] == "mlx" and len(parts) >= 2 and not parts[1][:1].isdigit():
        # noarch: e.g. "mlx-metal". Versioned (arch) MLX variants are not supported yet.
        return NoarchVariant(framework=MlxNoarch(), backend=Noarch.parse("-".join(parts[1:])))
    elif parts[0].startswith("torch"):
        # Legacy format: `torch210[-cxx11]-...`.
        legacy = True
        n = 2 if len(parts) >= 2 and parts[1] in ("cxx11", "cxx98") else 1
        framework = Torch.parse("-".join(parts[:n]), legacy=legacy)
    elif parts[0] == "tvm" and len(parts) >= 2 and parts[1].startswith("ffi"):
        # Legacy: `tvm-ffi01-...`, new: `tvm-ffi-0.1-...`.
        legacy = not (len(parts) > 2 and parts[1] == "ffi")
        n = 2 if legacy else 3
        framework = TvmFfi.parse("-".join(parts[:n]), legacy=legacy)
    else:
        raise ValueError(f"Unknown framework in variant string: {variant_str!r}")

    backend_parts, system = _split_system(parts[n:])
    backend_parts, python = _split_python(backend_parts, legacy=legacy)
    if python is None:
        raise ValueError(f"Missing Python version and ABI in variant string: {variant_str!r}")
    if not backend_parts:
        raise ValueError(f"Missing backend in variant string: {variant_str!r}")
    backend = parse_backend(backend_parts, legacy=legacy)
    return ArchVariant(framework=framework, backend=backend, python=python, system=system)


def parse_sdk_variant(variant_str: str) -> SdkVariant:
    """Parse a variant string into an SdkVariant."""
    parts = variant_str.split("-")

    if parts[0] != "sdk":
        raise ValueError(f"Not an SDK variant string: {variant_str!r}")
    rest = parts[1:]

    backend: Backend | Noarch | None = None
    python: Python | None = None
    system: System | None = None
    if _PYTHON_MARKER in rest:
        rest, system = _split_system(rest)
        rest, python = _split_python(rest, legacy=False)
        if rest:
            backend = parse_backend(rest, legacy=False)
    elif rest:
        # `sdk-<backend>`: no compiled parts, only the name of the backend.
        if len(rest) != 1:
            raise ValueError(f"Invalid SDK variant string: {variant_str!r}")
        backend = Noarch.parse(rest[0])

    return SdkVariant(backend=backend, python=python, system=system)


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
        mlx_available=has_mlx,
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
    mlx_available: bool = True,
) -> tuple[list[Variant], list[Decision]]:
    """Resolve the best matching variant given explicit system parameters.

    The Python version and free-threaded state default to a GIL-enabled
    Python 3.15, which supports abi3 and abi3t variants (up to Python 3.15).

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
        mlx_available,
    )
    trace = _sort_variants(trace)

    applicable = [decision.variant for decision in trace if isinstance(decision, VariantAccepted)]
    return applicable, trace


def _is_free_threaded_build() -> bool:
    """Check if the Python interpreter is a free-threaded build."""
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def _python_mismatch(python: Python, python_version: Version, free_threaded: bool) -> str | None:
    """Return the reason if the Python version or ABI is not supported by the
    interpreter."""
    if python.abi == "abi3" and free_threaded:
        # Free-threaded builds do not load abi3 extensions.
        return "abi3 is not supported on free-threaded Python, abi3t is required"
    if python_version < python.version:
        return f"Python {python.version} or later is required, system Python version is {python_version}"
    return None


def _system_mismatch(system: System, cpu: str, os: str) -> str | None:
    """Return the reason if the CPU or OS of the variant do not match the
    current system."""
    if system.platform != cpu:
        return f"CPU ({system.platform}) does not match system CPU ({cpu})"
    if system.os != os:
        return f"OS ({system.os}) does not match system OS ({os})"
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
    mlx_available: bool = True,
) -> list[Decision]:
    """Return only the variants applicable to the current system."""
    result: list[Decision] = []
    for v in variants:
        if isinstance(v, ArchVariant):
            # Skip variants with a Python ABI that the interpreter cannot load,
            # or with a non-matching CPU or OS.
            reason = _python_mismatch(v.python, python_version, free_threaded) or _system_mismatch(v.system, cpu, os)
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
            reason = _backend_mismatch(v.backend, selected_backend, macos_version)
            if reason is not None:
                result.append(VariantRejected(variant=v, reason=reason))
                continue
        elif isinstance(v, NoarchVariant):
            if isinstance(v.framework, MlxNoarch) and not mlx_available:
                result.append(VariantRejected(variant=v, reason="MLX is not installed in the environment"))
                continue
            # Only noarch variants with a matching backend or "universal"
            # are applicable.
            noarch_backend_name = selected_backend.name
            if v.backend.backend_name != noarch_backend_name and v.backend.backend_name != "universal":
                result.append(
                    VariantRejected(
                        variant=v,
                        reason=f"backend ({v.backend.backend_name}) does not match system backend ({noarch_backend_name}) and is not universal",
                    )
                )
                continue
        elif isinstance(v, SdkVariant):
            reason = None
            if v.python is not None:
                reason = _python_mismatch(v.python, python_version, free_threaded)
            if reason is None and v.system is not None:
                reason = _system_mismatch(v.system, cpu, os)
            if reason is None and v.backend is not None:
                if isinstance(v.backend, Noarch):
                    if v.backend.backend_name != selected_backend.name:
                        reason = f"backend ({v.backend.backend_name}) does not match system backend ({selected_backend.name})"
                else:
                    reason = _backend_mismatch(v.backend, selected_backend, macos_version)
            if reason is not None:
                result.append(VariantRejected(variant=v, reason=reason))
                continue
        result.append(VariantAccepted(variant=v))
    return result


def _python_order(python: Python) -> tuple[int, int, int]:
    """Sort key for the Python version and ABI: abi3 before abi3t, then higher
    Python versions first."""
    return (
        _PYTHON_ABIS.index(python.abi),
        -python.version.major,
        -python.version.minor,
    )


def _sort_variants(
    variants: list[Decision],
) -> list[Decision]:
    """Sort the decision trace in preference order:

    1. AcceptedVariant before RejectedVariant.
    2. Torch stable ABI arch kernels, with highest compatible version first,
       then highest compatible CUDA version.
    2. Torch arch kernels (tagless before C++ ABI-tagged) with the highest compatible CUDA version.
    3. tvm-ffi arch kernels with with the highest compatible CUDA version.
    4. SDK arch kernels with the highest compatible CUDA version, followed by
       backend-agnostic SDK arch kernels.
    5. Torch noarch kernels.
    6. Old Torch universal kernels.
    7. SDK kernels for a specific backend.
    8. SDK kernels that are framework and backend-agnostic.

    Arch kernels that are otherwise equal are ordered by Python ABI: abi3
    before abi3t, since GIL-enabled builds prefer abi3 (abi3 variants are never
    accepted on free-threaded builds). Then the highest Python version comes
    first (a higher Python version is preferred, like for wheel tags).
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
            if isinstance(v.backend, (CUDA, ROCm, XPU, CANN)):
                # Order by backend version in reverse (higher is better).
                backend_order = -v.backend.version.minor
            else:
                backend_order = 0
            return (decision_order, framework_order, *abi_version_order, backend_order, *_python_order(v.python))
        elif isinstance(v, SdkVariant):
            if v.python is not None:
                if v.backend is None:
                    # Backend-agnostic is less specific than any backend.
                    backend_order = 1
                elif isinstance(v.backend, (CUDA, ROCm, XPU, CANN)):
                    backend_order = -v.backend.version.minor
                else:
                    backend_order = 0
                return (decision_order, 3, 0, 0, backend_order, *_python_order(v.python))
            elif v.backend is not None:
                return (decision_order, 5, 0, 0, 0, 0, 0, 0)
            return (decision_order, 6, 0, 0, 0, 0, 0, 0)
        else:
            assert isinstance(v, NoarchVariant)
            universal_order = 1 if v.backend.backend_name == "universal" else 0
            return (decision_order, 4, 0, 0, universal_order, 0, 0, 0)

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
