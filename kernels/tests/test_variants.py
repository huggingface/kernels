import pytest
from huggingface_hub import HfApi
from packaging.version import Version

from kernels.backends import CPU, CUDA, Metal, ROCm
from kernels.variants import (
    Variant,
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
    "torch-tpu",
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


@pytest.fixture(params=["", "-cxx11"], ids=["tagless", "cxx11"])
def linux_abi(request) -> str:
    # Test build variants with and without the C++ ABI tag. kernel-builder
    # used to add an ABI tag to distingiush between the C++98 and C++11
    # ABIs.
    return request.param


def _resolve_variants(linux_abi: str) -> list[Variant]:
    return [
        parse_variant(s)
        for s in [
            f"torch210{linux_abi}-cu128-x86_64-linux",
            f"torch210{linux_abi}-cu126-x86_64-linux",
            f"torch210{linux_abi}-cu130-x86_64-linux",
            f"torch210{linux_abi}-rocm70-x86_64-linux",
            f"torch210{linux_abi}-cpu-x86_64-linux",
            "torch210-cpu-aarch64-darwin",
            "torch210-metal-aarch64-darwin",
            "torch-cuda",
            "torch-cpu",
        ]
    ]


def test_resolve_cuda_exact(linux_abi):
    # CUDA 12.8 should resolve to cu128.
    variants = _resolve_variants(linux_abi)
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
    assert result[0].variant_str == f"torch210{linux_abi}-cu128-x86_64-linux"
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(variants)


def test_resolve_cuda_best_older_minor(linux_abi):
    # CUDA 12.9 is not available, should fall back to cu128 (highest <= 12.9).
    variants = _resolve_variants(linux_abi)
    result, trace = _resolve_variant_for_system(
        variants=variants,
        selected_backend=CUDA(Version("12.9")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == f"torch210{linux_abi}-cu128-x86_64-linux"
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(variants)


def test_resolve_cuda_no_newer_minor(linux_abi):
    # CUDA 12.5 is older than all the variants, fall back to noarch.
    variants = _resolve_variants(linux_abi)
    result, trace = _resolve_variant_for_system(
        variants=variants,
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
    assert {vs.variant for vs in trace} == set(variants)


def test_resolve_cuda_no_different_major(linux_abi):
    # Different major version must not match.
    variants = _resolve_variants(linux_abi)
    result, trace = _resolve_variant_for_system(
        variants=variants,
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
    assert {vs.variant for vs in trace} == set(variants)


def test_resolve_rocm(linux_abi):
    variants = _resolve_variants(linux_abi)
    result, trace = _resolve_variant_for_system(
        variants=variants,
        selected_backend=ROCm(Version("7.0")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == f"torch210{linux_abi}-rocm70-x86_64-linux"
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(variants)


def test_resolve_cpu_linux(linux_abi):
    variants = _resolve_variants(linux_abi)
    result, trace = _resolve_variant_for_system(
        variants=variants,
        selected_backend=CPU(),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result != []
    assert result[0].variant_str == f"torch210{linux_abi}-cpu-x86_64-linux"
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(variants)


def test_resolve_cpu_darwin(linux_abi):
    variants = _resolve_variants(linux_abi)
    result, trace = _resolve_variant_for_system(
        variants=variants,
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
    assert {vs.variant for vs in trace} == set(variants)


def test_resolve_metal_darwin(linux_abi):
    variants = _resolve_variants(linux_abi)
    result, trace = _resolve_variant_for_system(
        variants=variants,
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
    assert {vs.variant for vs in trace} == set(variants)


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


def test_resolve_noarch_fallback(linux_abi):
    # With no matching arch variant, should fall back to torch noarch.
    variants = _resolve_variants(linux_abi)
    result, trace = _resolve_variant_for_system(
        variants=variants,
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
    assert {vs.variant for vs in trace} == set(variants)


def test_resolve_no_match(linux_abi):
    variants = _resolve_variants(linux_abi)
    result, trace = _resolve_variant_for_system(
        variants=variants,
        selected_backend=ROCm(Version("7.0")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.9"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result == []
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(variants)


def _resolve_variants_universal(linux_abi: str) -> list[Variant]:
    return [
        parse_variant(s)
        for s in [
            f"torch210{linux_abi}-cu128-x86_64-linux",
            "torch-universal",
        ]
    ]


def test_resolve_universal_matches_any_backend(linux_abi):
    # Universal works with every backend.
    variants = _resolve_variants_universal(linux_abi)
    result, trace = _resolve_variant_for_system(
        variants=variants,
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
    assert {vs.variant for vs in trace} == set(variants)


def test_resolve_universal_is_last_resort(linux_abi):
    # Specific match is preferred over universal.
    variants = _resolve_variants_universal(linux_abi)
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
    assert result[0].variant_str == f"torch210{linux_abi}-cu128-x86_64-linux"
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(variants)


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


def _resolve_variants_no_noarch(linux_abi: str) -> list[Variant]:
    return [
        parse_variant(s)
        for s in [
            f"torch210{linux_abi}-cu126-x86_64-linux",
            f"torch210{linux_abi}-cu128-x86_64-linux",
            f"torch210{linux_abi}-cu130-x86_64-linux",
        ]
    ]


def test_resolve_cuda_no_newer_minor_no_noarch(linux_abi):
    # No compatible variant for 12.5.
    variants = _resolve_variants_no_noarch(linux_abi)
    result, trace = _resolve_variant_for_system(
        variants=variants,
        selected_backend=CUDA(Version("12.5")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result == []
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(variants)


def test_resolve_cuda_no_different_major_no_noarch(linux_abi):
    # 11.8 has a different major, so there is no compatible fallback.
    variants = _resolve_variants_no_noarch(linux_abi)
    result, trace = _resolve_variant_for_system(
        variants=variants,
        selected_backend=CUDA(Version("11.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result == []
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(variants)


def _resolve_variants_stable_abi(linux_abi: str) -> list[Variant]:
    return [
        parse_variant(s)
        for s in [
            "torch-stable-abi211-cu128-x86_64-linux",
            f"torch210{linux_abi}-cu128-x86_64-linux",
            "torch-cuda",
        ]
    ]


def test_resolve_stable_abi_accepted(linux_abi):
    # Stable ABI 2.11 is accepted when torch_version == stable ABI version.
    variants = _resolve_variants_stable_abi(linux_abi)
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


def test_resolve_stable_abi_accepted_newer_torch(linux_abi):
    # Stable ABI 2.11 is also accepted when torch_version > stable ABI version.
    variants = _resolve_variants_stable_abi(linux_abi)
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


def test_resolve_stable_abi_rejected_newer_abi(linux_abi):
    # Stable ABI 2.11 is rejected when torch_version < stable ABI version.
    variants = _resolve_variants_stable_abi(linux_abi)
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
    assert result[0].variant_str == f"torch210{linux_abi}-cu128-x86_64-linux"
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(variants)


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


def test_resolve_mixed_abi_tags():
    # The legacy cxx98 tag is rejected on a cxx11 Torch, while both the
    # tagless and cxx11-tagged variants are accepted (tagless first).
    variants = [
        parse_variant(s)
        for s in [
            "torch210-cxx98-cu128-x86_64-linux",
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
    assert [v.variant_str for v in result] == [
        "torch210-cu128-x86_64-linux",
        "torch210-cxx11-cu128-x86_64-linux",
    ]
    rejected = {vs.variant.variant_str: vs.reason for vs in trace if isinstance(vs, VariantRejected)}
    assert "CXX11 ABI" in rejected["torch210-cxx98-cu128-x86_64-linux"]
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(variants)


def test_resolve_stable_abi_preferred_over_torch(linux_abi):
    # TorchStableAbi variant is preferred over a regular Torch variant of the same version.
    variants = [
        parse_variant(s)
        for s in [
            "torch-stable-abi211-cu128-x86_64-linux",
            f"torch211{linux_abi}-cu128-x86_64-linux",
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
