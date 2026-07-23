import pytest
from huggingface_hub import HfApi
from packaging.version import Version

from kernels.backends import CPU, CUDA, Metal, ROCm
from kernels.variants import (
    VariantAccepted,
    VariantRejected,
    _resolve_variant_for_system,
    get_variants,
    parse_variant,
)

VARIANT_STRINGS = (
    [
        f"{torch}{abi}-{backend}-{system}"
        for torch in ["torch25", "torch29", "torch210"]
        for abi in ["", "-cxx98", "-cxx11"]
        for backend in [
            "cpu",
            "cu126",
            "cu128",
            "cu130",
            "rocm63",
            "rocm64",
            "xpu20252",
        ]
        for system in ["aarch64-linux", "x86_64-linux"]
    ]
    + [
        f"{framework}-{backend}-{system}"
        for framework in ["torch25", "torch29", "torch210", "tvm-ffi01"]
        for backend in ["cpu", "metal"]
        for system in ["aarch64-darwin"]
    ]
    + [
        f"{tvmFfi}-{backend}-{system}"
        for tvmFfi in ["tvm-ffi01"]
        for backend in [
            "cpu",
            "cu126",
            "cu128",
            "cu130",
            "rocm63",
            "rocm64",
            "xpu20252",
        ]
        for system in ["aarch64-linux", "x86_64-linux"]
    ]
)

STABLE_ABI_VARIANT_STRINGS = [
    f"torch-stable-abi{abi_ver}-{backend}-{system}"
    for abi_ver in ["211", "29"]
    for backend in [
        "cpu",
        "cu126",
        "cu128",
        "cu130",
        "rocm63",
        "rocm64",
        "xpu20252",
    ]
    for system in ["aarch64-linux", "x86_64-linux"]
] + [
    f"torch-stable-abi{abi_ver}-{backend}-{system}"
    for abi_ver in ["211", "29"]
    for backend in ["cpu", "metal"]
    for system in ["aarch64-darwin"]
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
    "torch210-cpu-aarch64-darwin",
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
    "torch211-cpu-aarch64-darwin",
    "torch211-cxx11-cpu-aarch64-linux",
    "torch211-cxx11-cpu-x86_64-linux",
    "torch211-cxx11-cu126-aarch64-linux",
    "torch211-cxx11-cu126-x86_64-linux",
    "torch211-cxx11-cu128-aarch64-linux",
    "torch211-cxx11-cu128-x86_64-linux",
    "torch211-cxx11-cu130-aarch64-linux",
    "torch211-cxx11-cu130-x86_64-linux",
    "torch211-cxx11-rocm71-x86_64-linux",
    "torch211-cxx11-rocm72-x86_64-linux",
    "torch211-cxx11-xpu20253-x86_64-linux",
    "torch211-metal-aarch64-darwin",
    "torch212-cpu-aarch64-darwin",
    "torch212-cxx11-cpu-aarch64-linux",
    "torch212-cxx11-cpu-x86_64-linux",
    "torch212-cxx11-cu126-aarch64-linux",
    "torch212-cxx11-cu126-x86_64-linux",
    "torch212-cxx11-cu130-aarch64-linux",
    "torch212-cxx11-cu130-x86_64-linux",
    "torch212-cxx11-cu132-aarch64-linux",
    "torch212-cxx11-cu132-x86_64-linux",
    "torch212-cxx11-rocm71-x86_64-linux",
    "torch212-cxx11-rocm72-x86_64-linux",
    "torch212-cxx11-xpu20253-x86_64-linux",
    "torch212-metal-aarch64-darwin",
]


@pytest.mark.parametrize("variant_str", VARIANT_STRINGS)
def test_arch_variants(variant_str: str):
    # Roundtrip parse and generate variant string.
    assert parse_variant(variant_str).variant_str == variant_str


@pytest.mark.parametrize("variant_str", STABLE_ABI_VARIANT_STRINGS)
def test_stable_abi_variants(variant_str: str):
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
    result, trace = _resolve_variant_for_system(
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
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS)


def test_resolve_cuda_best_older_minor():
    # CUDA 12.9 is not available, should fall back to cu128 (highest <= 12.9).
    result, trace = _resolve_variant_for_system(
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
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS)


def test_resolve_cuda_no_newer_minor():
    # CUDA 12.5 is older than all the variants, fall back to noarch.
    result, trace = _resolve_variant_for_system(
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
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS)


def test_resolve_cuda_no_different_major():
    # Different major version must not match.
    result, trace = _resolve_variant_for_system(
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
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS)


def test_resolve_rocm():
    result, trace = _resolve_variant_for_system(
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
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS)


def test_resolve_cpu_linux():
    result, trace = _resolve_variant_for_system(
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
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS)


def test_resolve_cpu_darwin():
    result, trace = _resolve_variant_for_system(
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
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS)


def test_resolve_metal_darwin():
    result, trace = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS,
        selected_backend=Metal(),
        cpu="aarch64",
        os="darwin",
        torch_version=Version("2.10"),
        torch_cxx11_abi=None,
        tvm_ffi_version=None,
        macos_version=Version("26.0"),
    )
    assert result != []
    assert result[0].variant_str == "torch210-metal-aarch64-darwin"
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS)


RESOLVE_VARIANTS_METAL = [
    parse_variant(s)
    for s in [
        "torch210-cpu-aarch64-darwin",
        "torch210-metal-aarch64-darwin",
        "torch-metal",
    ]
]


@pytest.mark.parametrize("macos_version", [Version("15.7"), None])
def test_resolve_metal_darwin_old_macos(macos_version):
    # Metal arch kernels are built for macOS 26+, so they must be rejected
    # on older systems (falling back to the noarch variant).
    result, trace = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS_METAL,
        selected_backend=Metal(),
        cpu="aarch64",
        os="darwin",
        torch_version=Version("2.10"),
        torch_cxx11_abi=None,
        tvm_ffi_version=None,
        macos_version=macos_version,
    )
    assert result != []
    assert result[0].variant_str == "torch-metal"
    rejected = {vs.variant.variant_str: vs.reason for vs in trace if isinstance(vs, VariantRejected)}
    assert "require macOS 26.0" in rejected["torch210-metal-aarch64-darwin"]
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS_METAL)


def test_resolve_metal_darwin_new_macos():
    # On macOS 26+ the Metal arch kernel is accepted and preferred.
    result, trace = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS_METAL,
        selected_backend=Metal(),
        cpu="aarch64",
        os="darwin",
        torch_version=Version("2.10"),
        torch_cxx11_abi=None,
        tvm_ffi_version=None,
        macos_version=Version("26.1"),
    )
    assert result != []
    assert result[0].variant_str == "torch210-metal-aarch64-darwin"
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS_METAL)


def test_resolve_noarch_fallback():
    # With no matching arch variant, should fall back to torch noarch.
    result, trace = _resolve_variant_for_system(
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
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS)


def test_resolve_no_match():
    result, trace = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS,
        selected_backend=ROCm(Version("7.0")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.9"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result == []
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS)


RESOLVE_VARIANTS_UNIVERSAL = [
    parse_variant(s)
    for s in [
        "torch210-cxx11-cu128-x86_64-linux",
        "torch-universal",
    ]
]


def test_resolve_universal_matches_any_backend():
    # Universal works with every backend.
    result, trace = _resolve_variant_for_system(
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
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS_UNIVERSAL)


def test_resolve_universal_is_last_resort():
    # Specific match is preferred over universal.
    result, trace = _resolve_variant_for_system(
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
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS_UNIVERSAL)


def test_resolve_specific_noarch_preferred_over_universal():
    # Backend-specific noarch is preferred over universal.
    variants = [parse_variant(s) for s in ["torch-universal", "torch-cuda"]]
    result, trace = _resolve_variant_for_system(
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
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(variants)


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
    result, trace = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS_NO_NOARCH,
        selected_backend=CUDA(Version("12.5")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result == []
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS_NO_NOARCH)


def test_resolve_cuda_no_different_major_no_noarch():
    # 11.8 has a different major, so there is no compatible fallback.
    result, trace = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS_NO_NOARCH,
        selected_backend=CUDA(Version("11.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result == []
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS_NO_NOARCH)


RESOLVE_VARIANTS_STABLE_ABI = [
    parse_variant(s)
    for s in [
        "torch-stable-abi211-cu128-x86_64-linux",
        "torch210-cxx11-cu128-x86_64-linux",
        "torch-cuda",
    ]
]


def test_resolve_stable_abi_accepted():
    # Stable ABI 2.11 is accepted when torch_version == stable ABI version.
    result, trace = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS_STABLE_ABI,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.11"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch-stable-abi211-cu128-x86_64-linux"
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS_STABLE_ABI)


def test_resolve_stable_abi_accepted_newer_torch():
    # Stable ABI 2.11 is also accepted when torch_version > stable ABI version.
    result, trace = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS_STABLE_ABI,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.12"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch-stable-abi211-cu128-x86_64-linux"
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS_STABLE_ABI)


def test_resolve_stable_abi_rejected_newer_abi():
    # Stable ABI 2.11 is rejected when torch_version < stable ABI version.
    result, trace = _resolve_variant_for_system(
        variants=RESOLVE_VARIANTS_STABLE_ABI,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch210-cxx11-cu128-x86_64-linux"
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(RESOLVE_VARIANTS_STABLE_ABI)


def test_resolve_stable_abi_newest_version_preferred():
    # When multiple stable ABI versions are accepted, the newest is preferred.
    variants = [
        parse_variant(s)
        for s in [
            "torch-stable-abi29-cu128-x86_64-linux",
            "torch-stable-abi211-cu128-x86_64-linux",
        ]
    ]
    result, trace = _resolve_variant_for_system(
        variants=variants,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.12"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch-stable-abi211-cu128-x86_64-linux"
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(variants)


def test_resolve_tagless_preferred_over_abi_tagged():
    # Tagless variant (e.g. torch210-cu128) should be preferred over ABI-tagged
    # (e.g. torch210-cxx11-cu128) when both are accepted.
    variants = [
        parse_variant(s)
        for s in [
            "torch210-cxx11-cu128-x86_64-linux",
            "torch210-cu128-x86_64-linux",
        ]
    ]
    result, trace = _resolve_variant_for_system(
        variants=variants,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch210-cu128-x86_64-linux"
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(variants)


def test_resolve_stable_abi_preferred_over_torch():
    # TorchStableAbi variant is preferred over a regular Torch variant of the same version.
    variants = [
        parse_variant(s)
        for s in [
            "torch-stable-abi211-cu128-x86_64-linux",
            "torch211-cxx11-cu128-x86_64-linux",
        ]
    ]
    result, trace = _resolve_variant_for_system(
        variants=variants,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.11"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == "torch-stable-abi211-cu128-x86_64-linux"
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(variants)
