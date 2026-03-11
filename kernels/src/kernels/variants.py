import platform
import re

from packaging.version import parse

from kernels.backends import CUDA, Backend, _select_backend
from kernels.compat import has_torch, has_tvm_ffi

BUILD_VARIANT_REGEX = re.compile(
    r"^(torch\d+\d+|torch-(cpu|cuda|metal|neuron|rocm|xpu)|tvm-ffi\d+\d+)"
)


def _compatible_backend_variants(backend: Backend) -> list[str]:
    if isinstance(backend, CUDA):
        return [
            f"cu{backend.version.major}{minor}"
            for minor in range(backend.version.minor, -1, -1)
        ]
    return [backend.variant]


def _torch_build_variant(backend: str | None) -> list[str]:
    if not has_torch:
        return []

    selected_backend = _select_backend(backend)

    backend_variants = _compatible_backend_variants(selected_backend)

    import torch

    torch_version = parse(torch.__version__)
    cpu = platform.machine()
    os = platform.system().lower()

    if os == "darwin":
        cpu = "aarch64" if cpu == "arm64" else cpu
        return [
            f"torch{torch_version.major}{torch_version.minor}-{v}-{cpu}-{os}"
            for v in backend_variants
        ]
    elif os == "windows":
        cpu = "x86_64" if cpu == "AMD64" else cpu
        return [
            f"torch{torch_version.major}{torch_version.minor}-{v}-{cpu}-{os}"
            for v in backend_variants
        ]

    cxxabi = "cxx11" if torch.compiled_with_cxx11_abi() else "cxx98"
    return [
        f"torch{torch_version.major}{torch_version.minor}-{cxxabi}-{v}-{cpu}-{os}"
        for v in backend_variants
    ]


def _tvm_ffi_build_variant(backend: str | None) -> list[str]:
    if not has_tvm_ffi:
        return []

    selected_backend = _select_backend(backend)

    backend_variants = _compatible_backend_variants(selected_backend)

    import tvm_ffi

    tvm_ffi_version = parse(tvm_ffi.__version__)
    cpu = platform.machine()
    os = platform.system().lower()

    return [
        f"tvm-ffi{tvm_ffi_version.major}{tvm_ffi_version.minor}-{v}-{cpu}-{os}"
        for v in backend_variants
    ]


def _build_variant_noarch(backend: str | None) -> list[str]:
    selected_backend = _select_backend(backend)

    if selected_backend.name == "cann":
        return ["torch-npu"]
    else:
        return [f"torch-{selected_backend.name}"]


def _build_variant_universal() -> list[str]:
    # Once we support other frameworks, detection goes here.
    return ["torch-universal"] if has_torch else []


def _build_variants(backend: str | None) -> list[str]:
    """Return compatible build variants in preferred order."""
    return [
        *_torch_build_variant(backend),
        *_tvm_ffi_build_variant(backend),
        *_build_variant_noarch(backend),
        *_build_variant_universal(),
    ]
