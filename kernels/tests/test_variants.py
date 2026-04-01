import pytest
from huggingface_hub import HfApi
from packaging.version import Version

from kernels.backends import CPU, CUDA, ROCm
from kernels.variants import (
    _resolve_variant_for_system,
    get_variants,
    parse_variant,
    resolve_variants,
    system_variants,
)

VARIANT_STRINGS = [
    "torch25-cxx98-cu118-aarch64-linux",
    "torch25-cxx98-cpu-x86_64-linux",
    "torch29-cpu-aarch64-darwin",
    "torch29-cxx11-cpu-aarch64-linux",
    "torch29-cxx11-cpu-x86_64-linux",
    "torch29-cxx11-cu126-aarch64-linux",
    "torch29-cxx11-cu126-x86_64-linux",
    "torch29-cxx11-cu128-aarch64-linux",
    "torch29-cxx11-cu128-x86_64-linux",
    "torch29-cxx11-cu130-aarch64-linux",
    "torch29-cxx11-cu130-x86_64-linux",
    "torch29-cxx11-rocm63-x86_64-linux",
    "torch29-cxx11-rocm64-x86_64-linux",
    "torch29-cxx11-xpu20252-x86_64-linux",
    "torch29-cpu-aarch64-linux",
    "torch29-cpu-x86_64-linux",
    "torch29-cu126-aarch64-linux",
    "torch29-cu126-x86_64-linux",
    "torch29-cu128-aarch64-linux",
    "torch29-cu128-x86_64-linux",
    "torch29-cu130-aarch64-linux",
    "torch29-cu130-x86_64-linux",
    "torch29-rocm63-x86_64-linux",
    "torch29-rocm64-x86_64-linux",
    "torch29-xpu20252-x86_64-linux",
    "torch29-metal-aarch64-darwin",
    "torch210-cpu-aarch64-darwin",
    "torch210-cu128-x86_64-windows",
    "torch210-cxx11-cpu-aarch64-linux",
    "torch210-cxx11-cpu-x86_64-linux",
    "torch210-cxx11-cu126-aarch64-linux",
    "torch210-cxx11-cu126-x86_64-linux",
    "torch210-cxx11-cu128-aarch64-linux",
    "torch210-cxx11-cu128-x86_64-linux",
    "torch210-cxx11-cu130-aarch64-linux",
    "torch210-cxx11-cu130-x86_64-linux",
    "torch210-cxx11-rocm70-x86_64-linux",
    "torch210-cxx11-rocm71-x86_64-linux",
    "torch210-cxx11-xpu20253-x86_64-linux",
    "torch210-cpu-aarch64-linux",
    "torch210-cpu-x86_64-linux",
    "torch210-cu126-aarch64-linux",
    "torch210-cu126-x86_64-linux",
    "torch210-cu128-aarch64-linux",
    "torch210-cu128-x86_64-linux",
    "torch210-cu130-aarch64-linux",
    "torch210-cu130-x86_64-linux",
    "torch210-rocm70-x86_64-linux",
    "torch210-rocm71-x86_64-linux",
    "torch210-xpu20253-x86_64-linux",
    "torch210-metal-aarch64-darwin",
    "torch210-xpu20253-x86_64-windows",
    "tvm-ffi01-cpu-x86_64-linux",
    "tvm-ffi01-cu126-x86_64-linux",
    "tvm-ffi01-cu128-x86_64-linux",
    "tvm-ffi01-cu130-x86_64-linux",
    "tvm-ffi01-metal-aarch64-darwin",
    "tvm-ffi01-xpu20253-x86_64-linux",
]


NOARCH_VARIANT_STRINGS = [
    "torch-cpu",
    "torch-cuda",
    "torch-metal",
    "torch-neuron",
    "torch-rocm",
    "torch-xpu",
    "torch-npu",
    "torch-universal",
]

SUPERSET_VARIANT_STRINGS = [
    "torch29-cpu-aarch64-darwin",
    "torch29-cxx11-cpu-aarch64-linux",
    "torch29-cxx11-cpu-x86_64-linux",
    "torch29-cxx11-cu126-aarch64-linux",
    "torch29-cxx11-cu126-x86_64-linux",
    "torch29-cxx11-cu128-aarch64-linux",
    "torch29-cxx11-cu128-x86_64-linux",
    "torch29-cxx11-cu130-aarch64-linux",
    "torch29-cxx11-cu130-x86_64-linux",
    "torch29-cxx11-rocm63-x86_64-linux",
    "torch29-cxx11-rocm64-x86_64-linux",
    "torch29-cxx11-xpu20252-x86_64-linux",
    "torch29-metal-aarch64-darwin",
    "torch210-cpu-aarch64-darwin",
    "torch210-cu128-x86_64-windows",
    "torch210-cxx11-cpu-aarch64-linux",
    "torch210-cxx11-cpu-x86_64-linux",
    "torch210-cxx11-cu126-aarch64-linux",
    "torch210-cxx11-cu126-x86_64-linux",
    "torch210-cxx11-cu128-aarch64-linux",
    "torch210-cxx11-cu128-x86_64-linux",
    "torch210-cxx11-cu130-aarch64-linux",
    "torch210-cxx11-cu130-x86_64-linux",
    "torch210-cxx11-rocm70-x86_64-linux",
    "torch210-cxx11-rocm71-x86_64-linux",
    "torch210-cxx11-xpu20253-x86_64-linux",
    "torch210-metal-aarch64-darwin",
    "torch210-xpu20253-x86_64-windows",
]


@pytest.mark.parametrize("variant_str", VARIANT_STRINGS)
def test_arch_variants(variant_str: str):
    # Roundtrip parse and generate variant string.
    assert parse_variant(variant_str).variant_str == variant_str


@pytest.mark.parametrize("variant_str", NOARCH_VARIANT_STRINGS)
def test_noarch_variants(variant_str: str):
    # Roundtrip parse and generate variant string.
    assert parse_variant(variant_str).variant_str == variant_str


def test_get_variants():
    api = HfApi()
    variants = get_variants(api, repo_id="kernels-community/relu", revision="v1")
    variant_strs = {v.variant_str for v in variants}
    # Superset because new variants may be added in the future.
    assert variant_strs.issuperset(SUPERSET_VARIANT_STRINGS)


RESOLVE_VARIANTS = [
    parse_variant(s)
    for s in [
        "torch210-cxx11-cu128-x86_64-linux",
        "torch210-cxx11-cu126-x86_64-linux",
        "torch210-cxx11-cu130-x86_64-linux",
        "torch210-cxx11-rocm70-x86_64-linux",
        "torch210-cxx11-cpu-x86_64-linux",
        "torch210-cpu-aarch64-darwin",
        "torch210-metal-aarch64-darwin",
        "torch-cuda",
        "torch-cpu",
    ]
]


def test_resolve_cuda_exact():
    # CUDA 12.8 should resolve to cu128.
    result = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch210-cxx11-cu128-x86_64-linux"


def test_resolve_cuda_best_older_minor():
    # CUDA 12.9 is not available, should fall back to cu128 (highest <= 12.9).
    result = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS,
        selected_backend=CUDA(Version("12.9")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch210-cxx11-cu128-x86_64-linux"


def test_resolve_cuda_no_newer_minor():
    # CUDA 12.5 is older than all the variants, fall back to noarch.
    result = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS,
        selected_backend=CUDA(Version("12.5")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch-cuda"


def test_resolve_cuda_no_different_major():
    # Different major version must not match.
    result = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS,
        selected_backend=CUDA(Version("11.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch-cuda"


def test_resolve_rocm():
    result = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS,
        selected_backend=ROCm(Version("7.0")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch210-cxx11-rocm70-x86_64-linux"


def test_resolve_cpu_linux():
    result = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS,
        selected_backend=CPU(),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch210-cxx11-cpu-x86_64-linux"


def test_resolve_cpu_darwin():
    result = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS,
        selected_backend=CPU(),
        cpu="aarch64",
        os="darwin",
        torch_version=Version("2.10"),
        torch_cxx11_abi=None,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch210-cpu-aarch64-darwin"


def test_resolve_metal_darwin():
    result = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS,
        selected_backend=CPU(),
        cpu="aarch64",
        os="darwin",
        torch_version=Version("2.10"),
        torch_cxx11_abi=None,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch210-cpu-aarch64-darwin"


def test_resolve_noarch_fallback():
    # With no matching arch variant, should fall back to torch noarch.
    result = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS,
        selected_backend=CUDA(Version("12.8")),
        cpu="aarch64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch-cuda"


def test_resolve_no_match():
    result = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS,
        selected_backend=ROCm(Version("7.0")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.9"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result == []


RESOLVE_VARIANTS_UNIVERSAL = [
    parse_variant(s)
    for s in [
        "torch210-cxx11-cu128-x86_64-linux",
        "torch-universal",
    ]
]


def test_resolve_universal_matches_any_backend():
    # Universal works with every backend.
    result = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS_UNIVERSAL,
        selected_backend=ROCm(Version("7.0")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.9"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch-universal"


def test_resolve_universal_is_last_resort():
    # Specific match is preferred over universal.
    result = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS_UNIVERSAL,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch210-cxx11-cu128-x86_64-linux"


def test_resolve_specific_noarch_preferred_over_universal():
    # Backend-specific noarch is preferred over universal.
    variants = [parse_variant(s) for s in ["torch-universal", "torch-cuda"]]
    result = _resolve_variant_for_system(
        variants=variants,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.9"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch-cuda"


RESOLVE_VARIANTS_NO_NOARCH = [
    parse_variant(s)
    for s in [
        "torch210-cxx11-cu126-x86_64-linux",
        "torch210-cxx11-cu128-x86_64-linux",
        "torch210-cxx11-cu130-x86_64-linux",
    ]
]


def test_resolve_cuda_no_newer_minor_no_noarch():
    # No compatible variant for 12.5.
    result = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS_NO_NOARCH,
        selected_backend=CUDA(Version("12.5")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result == []


def test_resolve_cuda_no_different_major_no_noarch():
    # 11.8 has a different major, so there is no compatible fallback.
    result = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS_NO_NOARCH,
        selected_backend=CUDA(Version("11.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result == []


def test_system_variants_roundtrip():
    variants = system_variants()
    for v in variants:
        assert parse_variant(v.variant_str).variant_str == v.variant_str


def test_system_variants_no_duplicates():
    variants = system_variants()
    variant_strs = [v.variant_str for v in variants]
    assert len(variant_strs) == len(set(variant_strs))


def test_system_variants_all_resolve():
    variants = system_variants()
    resolved = resolve_variants(variants)
    assert set(v.variant_str for v in resolved) == set(v.variant_str for v in variants)
