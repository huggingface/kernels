import pytest
from huggingface_hub import HfApi
from packaging.version import Version

from kernels.backends import CANN, CPU, CUDA, XPU, Metal, ROCm, parse_backend
from kernels.variants import (
    ArchVariant,
    Noarch,
    Python,
    SdkVariant,
    System,
    Variant,
    VariantAccepted,
    VariantRejected,
    _resolve_variant_for_system,
    _split_python,
    _split_system,
    get_variants,
    parse_sdk_variant,
    parse_variant,
)

# Python versions and ABIs of new-style variants, `abi3t` requires Python 3.15.
PYTHON_VERSION_ABIS = ["cpython-3.9-abi3", "cpython-3.12-abi3", "cpython-3.15-abi3t"]

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
    "torch-cann",
    "torch-universal",
]

SDK_VARIANT_STRINGS = (
    ["sdk"]
    + [f"sdk-{backend}" for backend in ["cpu", "cuda", "metal", "rocm", "xpu", "cann"]]
    + [
        f"sdk-{backend}-{python_abi}-{system}"
        for backend in ["cpu", "cuda-12.6", "cuda-13.0", "rocm-6.3", "rocm-6.4", "xpu-2025.2", "cann-8.1"]
        for python_abi in PYTHON_VERSION_ABIS
        for system in ["aarch64-linux", "x86_64-linux"]
    ]
    + [
        f"sdk-{backend}-{python_abi}-aarch64-darwin"
        for backend in ["cpu", "metal"]
        for python_abi in PYTHON_VERSION_ABIS
    ]
    # Compiled, backend-agnostic.
    + [
        f"sdk-{python_abi}-{system}"
        for python_abi in PYTHON_VERSION_ABIS
        for system in ["aarch64-linux", "x86_64-linux", "aarch64-darwin"]
    ]
)

SUPERSET_VARIANT_STRINGS = [
    "torch210-cpu-aarch64-darwin",
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
    "torch211-cpu-aarch64-darwin",
    "torch211-cpu-aarch64-linux",
    "torch211-cpu-x86_64-linux",
    "torch211-cu126-aarch64-linux",
    "torch211-cu126-x86_64-linux",
    "torch211-cu128-aarch64-linux",
    "torch211-cu128-x86_64-linux",
    "torch211-cu130-aarch64-linux",
    "torch211-cu130-x86_64-linux",
    "torch211-rocm71-x86_64-linux",
    "torch211-rocm72-x86_64-linux",
    "torch211-xpu20253-x86_64-linux",
    "torch211-metal-aarch64-darwin",
    "torch212-cpu-aarch64-darwin",
    "torch212-cpu-aarch64-linux",
    "torch212-cpu-x86_64-linux",
    "torch212-cu126-aarch64-linux",
    "torch212-cu126-x86_64-linux",
    "torch212-cu130-aarch64-linux",
    "torch212-cu130-x86_64-linux",
    "torch212-cu132-aarch64-linux",
    "torch212-cu132-x86_64-linux",
    "torch212-rocm71-x86_64-linux",
    "torch212-rocm72-x86_64-linux",
    "torch212-xpu20253-x86_64-linux",
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


NEW_FORMAT_VARIANT_STRINGS = (
    [
        f"torch-{version}{abi}-{backend}-{python_abi}-{system}"
        for version in ["2.5", "2.10", "2.12"]
        for abi in ["", "-cxx98", "-cxx11"]
        for backend in ["cpu", "cuda-12.6", "cuda-13.0", "rocm-6.3", "rocm-10.0", "xpu-2025.2"]
        for python_abi in PYTHON_VERSION_ABIS
        for system in ["aarch64-linux", "x86_64-linux"]
    ]
    + [
        f"{framework}-{backend}-{python_abi}-aarch64-darwin"
        for framework in ["torch-2.10", "tvm-ffi-0.1"]
        for backend in ["cpu", "metal"]
        for python_abi in PYTHON_VERSION_ABIS
    ]
    + [
        f"tvm-ffi-0.1-{backend}-{python_abi}-{system}"
        for backend in ["cpu", "cuda-12.8", "rocm-6.4", "xpu-2025.2", "cann-8.1"]
        for python_abi in PYTHON_VERSION_ABIS
        for system in ["aarch64-linux", "x86_64-linux"]
    ]
    + [
        f"torch-stable-abi-{abi_ver}-{backend}-{python_abi}-{system}"
        for abi_ver in ["2.9", "2.11"]
        for backend in ["cpu", "cuda-12.8", "rocm-6.4", "xpu-2025.2"]
        for python_abi in PYTHON_VERSION_ABIS
        for system in ["aarch64-linux", "x86_64-linux"]
    ]
    + ["torch-2.10-cann-8.1-cpython-3.9-abi3-aarch64-linux"]
)


@pytest.mark.parametrize("variant_str", NEW_FORMAT_VARIANT_STRINGS)
def test_new_format_variants(variant_str: str):
    # Roundtrip parse and generate variant string.
    variant = parse_variant(variant_str)
    assert variant.variant_str == variant_str
    assert not variant.framework.legacy
    assert not variant.python.legacy
    assert f"-cpython-{variant.python.version}-{variant.python.abi}-" in variant_str


@pytest.mark.parametrize("variant_str", VARIANT_STRINGS + STABLE_ABI_VARIANT_STRINGS)
def test_legacy_variants_are_legacy(variant_str: str):
    variant = parse_variant(variant_str)
    assert variant.framework.legacy
    assert getattr(variant.backend, "legacy", True)
    assert variant.python.legacy
    # Legacy variants are always cpython-3.9-abi3.
    assert variant.python.abi == "abi3"
    assert variant.python.version == Version("3.9")


@pytest.mark.parametrize(
    "variant_str",
    [
        # Mixing formats is not allowed.
        "torch-2.10-cu128-cpython-3.9-abi3-x86_64-linux",
        "torch210-cuda-12.8-x86_64-linux",
        "torch210-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
        "torch-2.10-cxx11-rocm71-cpython-3.9-abi3-x86_64-linux",
        "torch210-cxx11-rocm-7.1-x86_64-linux",
        "torch-stable-abi-2.11-cu128-cpython-3.9-abi3-x86_64-linux",
        "torch-stable-abi211-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
        "tvm-ffi-0.1-xpu20253-cpython-3.9-abi3-x86_64-linux",
        "tvm-ffi01-xpu-2025.3-cpython-3.9-abi3-x86_64-linux",
        # Abbreviated backend name in the new format.
        "torch-2.10-cu-12.8-cpython-3.9-abi3-x86_64-linux",
        # Missing or superfluous backend version in the new format.
        "torch-2.10-cuda-cpython-3.9-abi3-x86_64-linux",
        "torch-2.10-cpu-12.8-cpython-3.9-abi3-x86_64-linux",
        # Malformed versions.
        "torch-2.-cpu-cpython-3.9-abi3-x86_64-linux",
        "torch-2.10.1-cpu-cpython-3.9-abi3-x86_64-linux",
        "torch-stable-abi-2-cpu-cpython-3.9-abi3-x86_64-linux",
        "tvm-ffi-0-cpu-cpython-3.9-abi3-x86_64-linux",
        # Missing Python ABI tag in the new format.
        "torch-2.10-cuda-12.8-x86_64-linux",
        "torch-2.10-cpu-x86_64-linux",
        "tvm-ffi-0.1-cpu-aarch64-darwin",
        "torch-stable-abi-2.11-cuda-12.8-x86_64-linux",
        # Unknown Python ABI tag, or the tag in the wrong place.
        "torch-2.10-cuda-12.8-abi4-x86_64-linux",
        "torch-2.10-cuda-12.8-abi3.abi3t-x86_64-linux",
        "torch-2.10-cuda-12.8-x86_64-linux-abi3",
        "torch-2.10-cpython-3.9-abi3-cuda-12.8-x86_64-linux",
        "torch-2.10-cuda-cpython-3.9-abi3-12.8-x86_64-linux",
        # The legacy format does not have a Python ABI tag.
        "torch210-cu128-cpython-3.9-abi3-x86_64-linux",
        "torch210-cu128-cpython-3.15-abi3t-x86_64-linux",
        "torch210-cxx11-cpu-cpython-3.9-abi3-x86_64-linux",
    ],
)
def test_variants_reject_mixed_or_malformed_formats(variant_str: str):
    with pytest.raises(ValueError):
        parse_variant(variant_str)


@pytest.mark.parametrize(
    "variant_str",
    [
        # New format.
        "torch-2.10-cpython-3.9-abi3-x86_64-linux",
        "torch-2.10-cxx11-cpython-3.9-abi3-x86_64-linux",
        "torch-stable-abi-2.11-cpython-3.9-abi3-x86_64-linux",
        "tvm-ffi-0.1-cpython-3.15-abi3t-x86_64-linux",
        # Legacy format.
        "torch210-x86_64-linux",
        "torch210-cxx11-x86_64-linux",
        "torch-stable-abi211-x86_64-linux",
        "tvm-ffi01-x86_64-linux",
    ],
)
def test_arch_variants_require_backend(variant_str: str):
    # Variants with compiled parts for a framework must have a backend.
    with pytest.raises(ValueError, match="Missing backend"):
        parse_variant(variant_str)


def test_new_format_variant_equals_legacy_variant():
    new = parse_variant("torch-2.10-cxx11-cuda-12.8-cpython-3.9-abi3-x86_64-linux")
    legacy = parse_variant("torch210-cxx11-cu128-x86_64-linux")
    assert new == legacy
    assert hash(new) == hash(legacy)
    assert new.variant_str == "torch-2.10-cxx11-cuda-12.8-cpython-3.9-abi3-x86_64-linux"
    assert legacy.variant_str == "torch210-cxx11-cu128-x86_64-linux"


def test_python_version_and_abi_take_part_in_equality():
    cp39 = parse_variant("torch-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux")
    cp312 = parse_variant("torch-2.10-cuda-12.8-cpython-3.12-abi3-x86_64-linux")
    abi3t = parse_variant("torch-2.10-cuda-12.8-cpython-3.15-abi3t-x86_64-linux")
    assert len({cp39, cp312, abi3t}) == 3
    assert cp39 != cp312
    assert cp39 != abi3t
    assert hash(cp39) != hash(cp312)
    assert cp39.python.abi == "abi3"
    assert cp312.python.version == Version("3.12")
    assert abi3t.python.abi == "abi3t"
    assert abi3t.python.version == Version("3.15")


def test_legacy_variant_is_cp39_abi3():
    # Legacy variants do not have a Python version and ABI, they are
    # implicitly cpython-3.9-abi3.
    legacy = parse_variant("torch210-cxx11-cu128-x86_64-linux")
    new = parse_variant("torch-2.10-cxx11-cuda-12.8-cpython-3.9-abi3-x86_64-linux")
    assert legacy == new
    assert hash(legacy) == hash(new)
    assert legacy != parse_variant("torch-2.10-cxx11-cuda-12.8-cpython-3.12-abi3-x86_64-linux")


def test_python_variant_str():
    # The Python version and ABI are implicit in the legacy format.
    assert Python().variant_str == ""
    assert Python(legacy=False).variant_str == "cpython-3.9-abi3"
    assert Python(version=Version("3.12"), legacy=False).variant_str == "cpython-3.12-abi3"
    assert Python(version=Version("3.15"), abi="abi3t", legacy=False).variant_str == "cpython-3.15-abi3t"


def test_system_variant_str():
    assert System(platform="x86_64", os="linux").variant_str == "x86_64-linux"
    assert System(platform="aarch64", os="darwin").variant_str == "aarch64-darwin"


def test_python_legacy_format_requires_cpython_3_9_abi3():
    with pytest.raises(ValueError):
        Python(version=Version("3.12"), legacy=True)
    with pytest.raises(ValueError):
        Python(version=Version("3.15"), abi="abi3t", legacy=True)


@pytest.mark.parametrize("python_version", ["3.9", "3.12", "3.14"])
def test_python_abi3t_requires_python_3_15(python_version: str):
    with pytest.raises(ValueError):
        Python(version=Version(python_version), abi="abi3t", legacy=False)


@pytest.mark.parametrize("python_version", ["3.15", "3.16"])
def test_python_abi3t_python_3_15_and_later(python_version: str):
    python = Python(version=Version(python_version), abi="abi3t", legacy=False)
    assert python.abi == "abi3t"


def test_split_system():
    # The system consists of the last two parts, in both formats.
    assert _split_system(["cu128", "x86_64", "linux"]) == (["cu128"], System(platform="x86_64", os="linux"))
    assert _split_system("cuda-12.8-cpython-3.9-abi3-aarch64-darwin".split("-")) == (
        ["cuda", "12.8", "cpython", "3.9", "abi3"],
        System(platform="aarch64", os="darwin"),
    )
    assert _split_system(["x86_64", "linux"]) == ([], System(platform="x86_64", os="linux"))
    with pytest.raises(ValueError):
        _split_system(["linux"])


def test_split_python():
    # In the new format the Python version and ABI start at the `cpython`
    # marker, so the number of parts before it does not matter.
    python = Python(version=Version("3.12"), abi="abi3", legacy=False)
    assert _split_python("cpython-3.12-abi3".split("-"), legacy=False) == ([], python)
    assert _split_python("cpu-cpython-3.12-abi3".split("-"), legacy=False) == (["cpu"], python)
    assert _split_python("cuda-12.8-cpython-3.12-abi3".split("-"), legacy=False) == (["cuda", "12.8"], python)
    # Without the marker there is no Python version and ABI.
    assert _split_python(["cuda"], legacy=False) == (["cuda"], None)
    assert _split_python([], legacy=False) == ([], None)
    # The marker must be followed by exactly a version and an ABI.
    with pytest.raises(ValueError):
        _split_python("cuda-12.8-cpython-3.12".split("-"), legacy=False)
    with pytest.raises(ValueError):
        _split_python("cuda-12.8-cpython-3.12-abi3-x".split("-"), legacy=False)
    # In the legacy format they are implicit.
    assert _split_python(["cu128"], legacy=True) == (["cu128"], Python())


@pytest.mark.parametrize("python_version", ["3.9", "3.10", "3.12", "3.13", "3.15", "3.16"])
def test_python_version(python_version: str):
    # Versions are dotted, `3.10` is Python 3.10 and not Python 3.1.
    variant = parse_variant(f"torch-2.10-cuda-12.8-cpython-{python_version}-abi3-x86_64-linux")
    assert variant.python.version == Version(python_version)
    assert variant.variant_str == f"torch-2.10-cuda-12.8-cpython-{python_version}-abi3-x86_64-linux"
    sdk = parse_variant(f"sdk-cpython-{python_version}-abi3-x86_64-linux")
    assert sdk.python.version == Version(python_version)
    assert sdk.variant_str == f"sdk-cpython-{python_version}-abi3-x86_64-linux"


@pytest.mark.parametrize(
    "variant_str",
    [
        # abi3t requires Python 3.15.
        "torch-2.10-cuda-12.8-cpython-3.9-abi3t-x86_64-linux",
        "torch-2.10-cuda-12.8-cpython-3.14-abi3t-x86_64-linux",
        "sdk-cpython-3.12-abi3t-x86_64-linux",
        # Missing Python version and ABI.
        "torch-2.10-cuda-12.8-abi3-x86_64-linux",
        "torch-2.10-cuda-12.8-3.9-abi3-x86_64-linux",
        "torch-2.10-cuda-12.8-cp39-abi3-x86_64-linux",
        "sdk-abi3-x86_64-linux",
        "sdk-3.12-abi3-x86_64-linux",
        # Missing ABI, or an incomplete system.
        "torch-2.10-cuda-12.8-cpython-3.9-x86_64-linux",
        "torch-2.10-cpu-cpython-3.15-x86_64-linux",
        "sdk-cpython-3.12-x86_64-linux",
        "sdk-cpython",
        "sdk-cpython-3.12",
        "sdk-cpython-3.12-abi3",
        "sdk-cpython-3.12-abi3-x86_64",
        # The Python version and ABI must be in order, followed by the platform
        # and OS.
        "torch-2.10-cuda-12.8-cpython-abi3-3.9-x86_64-linux",
        "torch-2.10-cuda-12.8-abi3-cpython-3.9-x86_64-linux",
        "torch-2.10-cuda-12.8-x86_64-linux-cpython-3.9-abi3",
        "sdk-abi3-cpython-3.12-x86_64-linux",
        # The system must be at the end and the marker can only appear once.
        "torch-2.10-cuda-12.8-cpython-3.9-abi3-cpython-3.9-abi3-x86_64-linux",
        "sdk-cpython-3.12-abi3-cuda-12.8-x86_64-linux",
        # Malformed Python versions.
        "torch-2.10-cuda-12.8-cpython-3-abi3-x86_64-linux",
        "torch-2.10-cuda-12.8-cpython-3.-abi3-x86_64-linux",
        "torch-2.10-cuda-12.8-cpython-.9-abi3-x86_64-linux",
        "torch-2.10-cuda-12.8-cpython-3.9.1-abi3-x86_64-linux",
        "torch-2.10-cuda-12.8-cpython-39-abi3-x86_64-linux",
        "torch-2.10-cuda-12.8-cpython-cp39-abi3-x86_64-linux",
        "torch-2.10-cuda-12.8-cpython-v3.9-abi3-x86_64-linux",
        "sdk-cpython--abi3-x86_64-linux",
        # Unknown ABI tags. Variants do not support extensions that are
        # compatible with both abi3 and abi3t.
        "torch-2.10-cuda-12.8-cpython-3.9-abi4-x86_64-linux",
        "torch-2.10-cuda-12.8-cpython-3.9-cp39-x86_64-linux",
        "torch-2.10-cuda-12.8-cpython-3.15-abi3.abi3t-x86_64-linux",
        "sdk-cpython-3.15-abi3.abi3t-x86_64-linux",
    ],
)
def test_variants_reject_malformed_python_version_or_abi(variant_str: str):
    with pytest.raises(ValueError):
        parse_variant(variant_str)


def test_arch_variant_backend_python_and_system():
    variant = parse_variant("torch-2.10-cxx11-cuda-12.8-cpython-3.15-abi3t-x86_64-linux")
    assert isinstance(variant, ArchVariant)
    assert variant.backend == CUDA(Version("12.8"))
    assert variant.python == Python(version=Version("3.15"), abi="abi3t", legacy=False)
    assert variant.system == System(platform="x86_64", os="linux")

    variant = parse_variant("torch210-cpu-aarch64-darwin")
    assert isinstance(variant, ArchVariant)
    assert variant.backend == CPU()
    assert variant.python == Python()
    assert variant.system == System(platform="aarch64", os="darwin")


@pytest.mark.parametrize(
    "legacy_str,new_str,backend",
    [
        ("cu128", "cuda-12.8", CUDA(Version("12.8"))),
        ("rocm71", "rocm-7.1", ROCm(Version("7.1"))),
        ("rocm714", "rocm-7.14", ROCm(Version("7.14"))),
        ("rocm100", "rocm-10.0", ROCm(Version("10.0"))),
        ("rocm200", "rocm-20.0", ROCm(Version("20.0"))),
        ("xpu20253", "xpu-2025.3", XPU(Version("2025.3"))),
        ("cann81", "cann-8.1", CANN(Version("8.1"))),
    ],
)
def test_parse_backend_formats(legacy_str: str, new_str: str, backend):
    legacy = parse_backend([legacy_str])
    new = parse_backend(new_str.split("-"), legacy=False)

    # The format does not take part in equality and hashing.
    assert legacy == new == backend
    assert hash(legacy) == hash(new)

    # Variant strings roundtrip in their own format.
    assert legacy.variant_str == legacy_str
    assert new.variant_str == new_str

    # The formats are not interchangeable.
    with pytest.raises(ValueError):
        parse_backend(new_str.split("-"))
    with pytest.raises(ValueError):
        parse_backend([legacy_str], legacy=False)


@pytest.mark.parametrize(
    "parts,legacy,backend",
    [
        (["cuda", "12.8"], False, CUDA(Version("12.8"))),
        (["rocm", "7.1"], False, ROCm(Version("7.1"))),
        (["xpu", "2025.3"], False, XPU(Version("2025.3"))),
        (["cpu"], False, CPU()),
        (["cu128"], True, CUDA(Version("12.8"))),
        (["cpu"], True, CPU()),
    ],
)
def test_parse_backend_parts(parts: list[str], legacy: bool, backend):
    # Backends are parsed from the parts of the variant string.
    parsed = parse_backend(parts, legacy=legacy)
    assert parsed == backend
    assert parsed.variant_str == "-".join(parts)


@pytest.mark.parametrize(
    "parts,legacy",
    [
        ([], False),
        ([], True),
        (["cuda"], False),
        (["cuda", "12.8", "extra"], False),
        (["cpu", "12.8"], False),
        (["cu128", "extra"], True),
        (["cuda", "12.8"], True),
    ],
)
def test_parse_backend_invalid_parts(parts: list[str], legacy: bool):
    with pytest.raises(ValueError):
        parse_backend(parts, legacy=legacy)


@pytest.mark.parametrize("backend_str", ["cpu", "metal", "neuron", "tpu"])
def test_parse_unversioned_backend_formats(backend_str: str):
    assert parse_backend([backend_str]).variant_str == backend_str
    assert parse_backend([backend_str], legacy=False).variant_str == backend_str


def test_new_format_resolves_legacy_ambiguity():
    # The legacy format splits greedily: the last digit is the minor
    # version. The new format can express larger minor versions.
    assert parse_backend(["cu1300"]).version == Version("130.0")
    assert parse_backend(["cuda", "13.0"], legacy=False).version == Version("13.0")
    assert parse_backend(["cuda", "12.10"], legacy=False).version == Version("12.10")


def test_resolve_new_format_cuda():
    variants = [
        parse_variant(s)
        for s in [
            "torch-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
            "torch-2.10-cuda-12.6-cpython-3.9-abi3-x86_64-linux",
            "torch-2.10-cuda-13.0-cpython-3.9-abi3-x86_64-linux",
            "torch-2.10-cpu-cpython-3.9-abi3-x86_64-linux",
        ]
    ]
    result, _ = _resolve_variant_for_system(
        variants=variants,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert result[0].variant_str == "torch-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux"


def test_resolve_new_format_other_backend():
    # Backends other than CUDA/ROCm are compared by equality, regardless of
    # the notation used in the variant.
    variants = [
        parse_variant("torch-2.10-xpu-2025.3-cpython-3.9-abi3-x86_64-linux"),
        parse_variant("torch210-xpu20252-x86_64-linux"),
    ]
    result, trace = _resolve_variant_for_system(
        variants=variants,
        selected_backend=XPU(Version("2025.3")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert [v.variant_str for v in result] == ["torch-2.10-xpu-2025.3-cpython-3.9-abi3-x86_64-linux"]
    rejected = {vs.variant.variant_str for vs in trace if isinstance(vs, VariantRejected)}
    assert rejected == {"torch210-xpu20252-x86_64-linux"}


def test_resolve_noarch_cann():
    # Noarch variants use the backend name, which is `cann` for CANN.
    variants = [parse_variant("torch-cann"), parse_variant("torch-npu")]
    result, trace = _resolve_variant_for_system(
        variants=variants,
        selected_backend=CANN(Version("8.1")),
        cpu="aarch64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )
    assert [v.variant_str for v in result] == ["torch-cann"]
    rejected = {vs.variant.variant_str for vs in trace if isinstance(vs, VariantRejected)}
    assert rejected == {"torch-npu"}


def _resolve_python_abi(variants: list[str], *, python_version: str, free_threaded: bool):
    parsed = [parse_variant(s) for s in variants]
    result, trace = _resolve_variant_for_system(
        variants=parsed,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
        python_version=Version(python_version),
        free_threaded=free_threaded,
    )
    rejected = {vs.variant.variant_str: vs.reason for vs in trace if isinstance(vs, VariantRejected)}
    return [v.variant_str for v in result], rejected


PYTHON_ABI_VARIANTS = [
    "torch-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
    "torch-2.10-cuda-12.8-cpython-3.15-abi3t-x86_64-linux",
    "torch210-cu128-x86_64-linux",
    "torch-cuda",
]


@pytest.mark.parametrize("python_version", ["3.10", "3.14"])
def test_resolve_python_abi_gil_before_3_15(python_version: str):
    # GIL-enabled builds before 3.15 cannot load abi3t.
    result, rejected = _resolve_python_abi(PYTHON_ABI_VARIANTS, python_version=python_version, free_threaded=False)
    assert result == [
        "torch-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
        "torch210-cu128-x86_64-linux",
        "torch-cuda",
    ]
    assert set(rejected) == {"torch-2.10-cuda-12.8-cpython-3.15-abi3t-x86_64-linux"}
    assert (
        f"Python 3.15 or later is required, system Python version is {python_version}"
        in rejected["torch-2.10-cuda-12.8-cpython-3.15-abi3t-x86_64-linux"]
    )


@pytest.mark.parametrize("python_version", ["3.15", "3.16"])
def test_resolve_python_abi_gil_3_15_and_later(python_version: str):
    # GIL-enabled builds load both. abi3 is preferred, the legacy variant is
    # abi3 as well. The variants only differ by Python ABI and notation.
    result, rejected = _resolve_python_abi(PYTHON_ABI_VARIANTS, python_version=python_version, free_threaded=False)
    assert result[-1] == "torch-cuda"
    assert set(result[:-1]) == {
        "torch-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
        "torch210-cu128-x86_64-linux",
        "torch-2.10-cuda-12.8-cpython-3.15-abi3t-x86_64-linux",
    }
    assert result[2] == "torch-2.10-cuda-12.8-cpython-3.15-abi3t-x86_64-linux"
    assert rejected == {}


@pytest.mark.parametrize("python_version", ["3.15", "3.16"])
def test_resolve_python_abi_free_threaded(python_version: str):
    # Free-threaded builds cannot load abi3, including the legacy variants,
    # which are abi3.
    result, rejected = _resolve_python_abi(PYTHON_ABI_VARIANTS, python_version=python_version, free_threaded=True)
    assert result == ["torch-2.10-cuda-12.8-cpython-3.15-abi3t-x86_64-linux", "torch-cuda"]
    assert set(rejected) == {"torch-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux", "torch210-cu128-x86_64-linux"}
    for reason in rejected.values():
        assert "abi3 is not supported on free-threaded Python" in reason


def test_resolve_python_abi_free_threaded_before_3_15():
    # Neither ABI is available, only noarch variants can be used.
    result, rejected = _resolve_python_abi(PYTHON_ABI_VARIANTS, python_version="3.14", free_threaded=True)
    assert result == ["torch-cuda"]
    assert len(rejected) == 3


PYTHON_VERSION_VARIANTS = [
    "torch-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
    "torch-2.10-cuda-12.8-cpython-3.12-abi3-x86_64-linux",
    "torch-2.10-cuda-12.8-cpython-3.15-abi3t-x86_64-linux",
    "torch-2.10-cuda-12.8-cpython-3.16-abi3t-x86_64-linux",
]


@pytest.mark.parametrize(
    "python_version,free_threaded,expected",
    [
        ("3.9", False, ["cpython-3.9-abi3"]),
        ("3.11", False, ["cpython-3.9-abi3"]),
        ("3.12", False, ["cpython-3.12-abi3", "cpython-3.9-abi3"]),
        ("3.14", False, ["cpython-3.12-abi3", "cpython-3.9-abi3"]),
        ("3.15", False, ["cpython-3.12-abi3", "cpython-3.9-abi3", "cpython-3.15-abi3t"]),
        ("3.16", False, ["cpython-3.12-abi3", "cpython-3.9-abi3", "cpython-3.16-abi3t", "cpython-3.15-abi3t"]),
        # Free-threaded builds only load abi3t, from 3.15.
        ("3.14", True, []),
        ("3.15", True, ["cpython-3.15-abi3t"]),
        ("3.16", True, ["cpython-3.16-abi3t", "cpython-3.15-abi3t"]),
    ],
)
def test_resolve_python_versions(python_version: str, free_threaded: bool, expected: list[str]):
    result, rejected = _resolve_python_abi(
        PYTHON_VERSION_VARIANTS, python_version=python_version, free_threaded=free_threaded
    )
    assert result == [f"torch-2.10-cuda-12.8-{tag}-x86_64-linux" for tag in expected]
    assert set(result) | set(rejected) == set(PYTHON_VERSION_VARIANTS)


def test_resolve_python_version_rejection_reason():
    _, rejected = _resolve_python_abi(PYTHON_VERSION_VARIANTS, python_version="3.11", free_threaded=False)
    assert (
        "Python 3.12 or later is required, system Python version is 3.11"
        in rejected["torch-2.10-cuda-12.8-cpython-3.12-abi3-x86_64-linux"]
    )


def test_resolve_python_version_does_not_outrank_backend_version():
    # The Python version only orders otherwise equal variants.
    variants = [
        "torch-2.10-cuda-12.6-cpython-3.12-abi3-x86_64-linux",
        "torch-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
    ]
    result, _ = _resolve_python_abi(variants, python_version="3.13", free_threaded=False)
    assert result == list(reversed(variants))


def test_resolve_legacy_variant_is_cp39_abi3():
    variants = ["torch210-cu128-x86_64-linux"]
    result, _ = _resolve_python_abi(variants, python_version="3.9", free_threaded=False)
    assert result == variants
    result, rejected = _resolve_python_abi(variants, python_version="3.9", free_threaded=True)
    assert result == []
    assert "abi3 is not supported on free-threaded Python" in rejected[variants[0]]


def test_resolve_python_abi_preference():
    # Otherwise equal variants: abi3 is preferred over abi3t, but the Python
    # ABI does not outrank the backend version.
    variants = [
        "torch-2.10-cuda-12.8-cpython-3.15-abi3t-x86_64-linux",
        "torch-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
        "torch-2.10-cuda-12.6-cpython-3.9-abi3-x86_64-linux",
        "torch-2.10-cuda-12.6-cpython-3.15-abi3t-x86_64-linux",
    ]
    result, _ = _resolve_python_abi(variants, python_version="3.15", free_threaded=False)
    assert result == [
        "torch-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
        "torch-2.10-cuda-12.8-cpython-3.15-abi3t-x86_64-linux",
        "torch-2.10-cuda-12.6-cpython-3.9-abi3-x86_64-linux",
        "torch-2.10-cuda-12.6-cpython-3.15-abi3t-x86_64-linux",
    ]


@pytest.mark.parametrize("variant_str", SDK_VARIANT_STRINGS)
def test_sdk_variants(variant_str: str):
    # Roundtrip parse and generate variant string.
    variant = parse_sdk_variant(variant_str)
    assert variant.variant_str == variant_str
    if variant.python is not None:
        assert not variant.python.legacy
        assert f"-cpython-{variant.python.version}-{variant.python.abi}-" in variant_str
    if variant.backend is not None and not isinstance(variant.backend, Noarch):
        assert not getattr(variant.backend, "legacy", False)


def test_sdk_variant_forms():
    # No backend, no compiled parts.
    assert parse_sdk_variant("sdk") == SdkVariant(backend=None, python=None, system=None)
    # Backend name only, no compiled parts.
    assert parse_sdk_variant("sdk-cuda") == SdkVariant(backend=Noarch(backend_name="cuda"), python=None, system=None)
    assert parse_sdk_variant("sdk-cpu") == SdkVariant(backend=Noarch(backend_name="cpu"), python=None, system=None)
    # Compiled, no backend.
    assert parse_sdk_variant("sdk-cpython-3.9-abi3-x86_64-linux") == SdkVariant(
        backend=None, python=Python(), system=System(platform="x86_64", os="linux")
    )
    assert parse_sdk_variant("sdk-cpython-3.15-abi3t-aarch64-darwin") == SdkVariant(
        backend=None,
        python=Python(version=Version("3.15"), abi="abi3t", legacy=False),
        system=System(platform="aarch64", os="darwin"),
    )
    # Compiled, with backend.
    assert parse_sdk_variant("sdk-cuda-12.8-cpython-3.9-abi3-x86_64-linux") == SdkVariant(
        backend=CUDA(version=Version("12.8")),
        python=Python(),
        system=System(platform="x86_64", os="linux"),
    )
    assert parse_sdk_variant("sdk-cuda-12.8-cpython-3.15-abi3t-x86_64-linux") == SdkVariant(
        backend=CUDA(version=Version("12.8")),
        python=Python(version=Version("3.15"), abi="abi3t", legacy=False),
        system=System(platform="x86_64", os="linux"),
    )
    assert parse_sdk_variant("sdk-cpu-cpython-3.9-abi3-x86_64-linux") == SdkVariant(
        backend=CPU(),
        python=Python(),
        system=System(platform="x86_64", os="linux"),
    )
    assert parse_sdk_variant("sdk-cuda-12.8-cpython-3.9-abi3-x86_64-linux") != parse_sdk_variant(
        "sdk-cuda-12.8-cpython-3.15-abi3t-x86_64-linux"
    )


def test_sdk_variant_backend_python_and_system_are_consistent():
    python = Python(legacy=False)
    system = System(platform="x86_64", os="linux")
    # Name-only backends have no compiled parts.
    with pytest.raises(ValueError):
        SdkVariant(backend=Noarch(backend_name="cuda"), python=python, system=system)
    # Backends with a version require compiled parts.
    with pytest.raises(ValueError):
        SdkVariant(backend=CUDA(Version("12.8")), python=None, system=None)
    with pytest.raises(ValueError):
        SdkVariant(backend=CPU(), python=None, system=None)
    # The Python version and ABI and the system go together.
    with pytest.raises(ValueError):
        SdkVariant(backend=None, python=python, system=None)
    with pytest.raises(ValueError):
        SdkVariant(backend=None, python=None, system=system)


@pytest.mark.parametrize(
    "variant_str",
    [
        # Wrong number of parts.
        "sdk-cuda-12.8",
        "sdk-cuda-12.8-abi3",
        "sdk-cuda-12.8-cpython-3.9-abi3-x86_64",
        "sdk-cuda-12.8-cpython-3.9-abi3-x86_64-linux-extra",
        "sdk-cpu-cpython-3.9-abi3-x86_64",
        "sdk-cpython-3.9-abi3-x86_64",
        "sdk-cpython-3.9-abi3-x86_64-linux-extra",
        # Missing Python ABI tag.
        "sdk-cuda-12.8-x86_64-linux",
        "sdk-cpu-x86_64-linux",
        "sdk-x86_64-linux",
        # Unknown Python ABI tag, or the tag in the wrong place.
        "sdk-cuda-12.8-abi4-x86_64-linux",
        "sdk-cuda-12.8-x86_64-linux-abi3",
        "sdk-cpython-3.9-abi3-cuda-12.8-x86_64-linux",
        # SDK variants only use the new format.
        "sdk-cu128-cpython-3.9-abi3-x86_64-linux",
        "sdk-rocm63-cpython-3.9-abi3-x86_64-linux",
        "sdk-xpu20252-cpython-3.9-abi3-x86_64-linux",
        # Missing, superfluous or malformed backend version.
        "sdk-cuda-cpython-3.9-abi3-x86_64-linux",
        "sdk-cpu-12.8-cpython-3.9-abi3-x86_64-linux",
        "sdk-cuda-12-cpython-3.9-abi3-x86_64-linux",
        # Unknown backend in arch form.
        "sdk-unknown-cpython-3.9-abi3-x86_64-linux",
        # Not an SDK variant.
        "torch-cpu",
        "torch-2.10-cpu-cpython-3.9-abi3-x86_64-linux",
        "sdkfoo",
        "",
    ],
)
def test_sdk_variants_invalid(variant_str: str):
    with pytest.raises(ValueError):
        parse_sdk_variant(variant_str)


@pytest.mark.parametrize("variant_str", SDK_VARIANT_STRINGS)
def test_parse_variant_sdk(variant_str: str):
    # SDK variants are parsed by the generic parser, so that they can be
    # resolved like other variants.
    variant = parse_variant(variant_str)
    assert isinstance(variant, SdkVariant)
    assert variant == parse_sdk_variant(variant_str)
    assert variant.variant_str == variant_str


SDK_RESOLVE_VARIANTS = [
    parse_variant(s)
    for s in [
        "sdk",
        "sdk-cuda",
        "sdk-cpu",
        "sdk-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
        "sdk-cuda-12.8-cpython-3.15-abi3t-x86_64-linux",
        "sdk-cuda-12.6-cpython-3.9-abi3-x86_64-linux",
        "sdk-cuda-13.0-cpython-3.9-abi3-x86_64-linux",
        "sdk-cuda-12.8-cpython-3.9-abi3-aarch64-linux",
        "sdk-rocm-6.3-cpython-3.9-abi3-x86_64-linux",
        "sdk-metal-cpython-3.9-abi3-aarch64-darwin",
        "sdk-cpython-3.9-abi3-x86_64-linux",
        "sdk-cpython-3.15-abi3t-x86_64-linux",
        "sdk-cpython-3.9-abi3-aarch64-linux",
        "torch-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
    ]
]


def test_resolve_sdk_without_framework():
    # SDK variants do not need a framework to be installed.
    result, trace = _resolve_variant_for_system(
        variants=SDK_RESOLVE_VARIANTS,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=None,
        torch_cxx11_abi=None,
        tvm_ffi_version=None,
    )
    # Highest compatible CUDA version first (abi3 before abi3t), then backend
    # SDKs, then the backend-agnostic SDK.
    assert [v.variant_str for v in result] == [
        "sdk-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
        "sdk-cuda-12.8-cpython-3.15-abi3t-x86_64-linux",
        "sdk-cuda-12.6-cpython-3.9-abi3-x86_64-linux",
        "sdk-cpython-3.9-abi3-x86_64-linux",
        "sdk-cpython-3.15-abi3t-x86_64-linux",
        "sdk-cuda",
        "sdk",
    ]
    assert result == [vs.variant for vs in trace if isinstance(vs, VariantAccepted)]
    assert {vs.variant for vs in trace} == set(SDK_RESOLVE_VARIANTS)

    rejected = {vs.variant.variant_str: vs.reason for vs in trace if isinstance(vs, VariantRejected)}
    assert set(rejected) == {
        "sdk-cpu",
        "sdk-cuda-13.0-cpython-3.9-abi3-x86_64-linux",
        "sdk-cuda-12.8-cpython-3.9-abi3-aarch64-linux",
        "sdk-rocm-6.3-cpython-3.9-abi3-x86_64-linux",
        "sdk-metal-cpython-3.9-abi3-aarch64-darwin",
        "sdk-cpython-3.9-abi3-aarch64-linux",
        "torch-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
    }
    assert "does not match system CPU" in rejected["sdk-cpython-3.9-abi3-aarch64-linux"]
    assert "does not match system backend" in rejected["sdk-cpu"]
    assert "CUDA version (13.0)" in rejected["sdk-cuda-13.0-cpython-3.9-abi3-x86_64-linux"]
    assert "does not match system CPU" in rejected["sdk-cuda-12.8-cpython-3.9-abi3-aarch64-linux"]
    assert "does not match selected backend" in rejected["sdk-rocm-6.3-cpython-3.9-abi3-x86_64-linux"]


def test_resolve_sdk_preference_order():
    # Framework-specific arch variants are preferred over SDK variants, SDK
    # arch variants (backend-specific before backend-agnostic) over Torch
    # noarch, and the backend-agnostic SDK is last.
    expected = [
        "torch-stable-abi-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
        "torch-2.10-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
        "tvm-ffi-0.1-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
        "sdk-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
        "sdk-cpython-3.9-abi3-x86_64-linux",
        "torch-cuda",
        "torch-universal",
        "sdk-cuda",
        "sdk",
    ]
    variants = [parse_variant(s) for s in reversed(expected)]
    result, _ = _resolve_variant_for_system(
        variants=variants,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.10"),
        torch_cxx11_abi=True,
        tvm_ffi_version=Version("0.1"),
    )
    assert [v.variant_str for v in result] == expected


@pytest.mark.parametrize(
    "python_version,free_threaded,expected",
    [
        # GIL builds before 3.15 cannot load abi3t.
        (
            "3.14",
            False,
            ["sdk-cuda-12.8-cpython-3.9-abi3-x86_64-linux", "sdk-cpython-3.9-abi3-x86_64-linux", "sdk-cuda", "sdk"],
        ),
        # GIL builds from 3.15 load both, abi3 is preferred.
        (
            "3.15",
            False,
            [
                "sdk-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
                "sdk-cuda-12.8-cpython-3.15-abi3t-x86_64-linux",
                "sdk-cpython-3.9-abi3-x86_64-linux",
                "sdk-cpython-3.15-abi3t-x86_64-linux",
                "sdk-cuda",
                "sdk",
            ],
        ),
        # Free-threaded builds cannot load abi3.
        (
            "3.15",
            True,
            [
                "sdk-cuda-12.8-cpython-3.15-abi3t-x86_64-linux",
                "sdk-cpython-3.15-abi3t-x86_64-linux",
                "sdk-cuda",
                "sdk",
            ],
        ),
        # Variants without compiled parts do not depend on the Python ABI.
        ("3.14", True, ["sdk-cuda", "sdk"]),
    ],
)
def test_resolve_sdk_python_abi(python_version: str, free_threaded: bool, expected: list[str]):
    variants = [
        parse_variant(s)
        for s in [
            "sdk",
            "sdk-cuda",
            "sdk-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
            "sdk-cuda-12.8-cpython-3.15-abi3t-x86_64-linux",
            "sdk-cpython-3.9-abi3-x86_64-linux",
            "sdk-cpython-3.15-abi3t-x86_64-linux",
        ]
    ]
    result, _ = _resolve_variant_for_system(
        variants=variants,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=None,
        torch_cxx11_abi=None,
        tvm_ffi_version=None,
        python_version=Version(python_version),
        free_threaded=free_threaded,
    )
    assert [v.variant_str for v in result] == expected


@pytest.mark.parametrize(
    "python_version,free_threaded,expected",
    [
        # Before 3.12 only the SDK with a Python 3.9 minimum is usable.
        ("3.11", False, ["sdk-cuda-12.8-cpython-3.9-abi3-x86_64-linux", "sdk-cuda"]),
        # From 3.12 the newest compatible Python version is preferred.
        (
            "3.13",
            False,
            [
                "sdk-cuda-12.8-cpython-3.12-abi3-x86_64-linux",
                "sdk-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
                "sdk-cpython-3.12-abi3-x86_64-linux",
                "sdk-cuda",
            ],
        ),
        # Free-threaded builds do not load abi3, only the SDK without compiled
        # parts remains.
        ("3.15", True, ["sdk-cuda"]),
    ],
)
def test_resolve_sdk_python_version(python_version: str, free_threaded: bool, expected: list[str]):
    # SDKs can have a newer minimum Python version than kernels (e.g. nanobind
    # supports abi3 from Python 3.12).
    variants = [
        parse_variant(s)
        for s in [
            "sdk-cpython-3.12-abi3-x86_64-linux",
            "sdk-cuda-12.8-cpython-3.12-abi3-x86_64-linux",
            "sdk-cuda-12.8-cpython-3.9-abi3-x86_64-linux",
            "sdk-cuda",
        ]
    ]
    result, _ = _resolve_variant_for_system(
        variants=variants,
        selected_backend=CUDA(Version("12.8")),
        cpu="x86_64",
        os="linux",
        torch_version=None,
        torch_cxx11_abi=None,
        tvm_ffi_version=None,
        python_version=Version(python_version),
        free_threaded=free_threaded,
    )
    assert [v.variant_str for v in result] == expected


@pytest.mark.parametrize(
    "selected_backend", [CUDA(Version("12.8")), ROCm(Version("7.1")), CPU(), XPU(Version("2025.3"))]
)
def test_resolve_sdk_without_backend_matches_any_backend(selected_backend):
    # SDKs with compiled parts but without a backend work with any backend,
    # but the system (CPU, OS and Python ABI) must still match.
    variants = [
        parse_variant("sdk-cpython-3.9-abi3-x86_64-linux"),
        parse_variant("sdk-cpython-3.9-abi3-aarch64-linux"),
    ]
    result, trace = _resolve_variant_for_system(
        variants=variants,
        selected_backend=selected_backend,
        cpu="x86_64",
        os="linux",
        torch_version=None,
        torch_cxx11_abi=None,
        tvm_ffi_version=None,
    )
    assert [v.variant_str for v in result] == ["sdk-cpython-3.9-abi3-x86_64-linux"]
    rejected = {vs.variant.variant_str: vs.reason for vs in trace if isinstance(vs, VariantRejected)}
    assert "does not match system CPU" in rejected["sdk-cpython-3.9-abi3-aarch64-linux"]


@pytest.mark.parametrize("macos_version,accepted", [(None, False), (Version("25.4"), False), (Version("26.0"), True)])
def test_resolve_sdk_metal_macos(macos_version, accepted: bool):
    variants = [parse_variant("sdk-metal-cpython-3.9-abi3-aarch64-darwin")]
    result, trace = _resolve_variant_for_system(
        variants=variants,
        selected_backend=Metal(),
        cpu="aarch64",
        os="darwin",
        torch_version=None,
        torch_cxx11_abi=None,
        tvm_ffi_version=None,
        macos_version=macos_version,
    )
    assert (result == variants) == accepted
    if not accepted:
        assert isinstance(trace[0], VariantRejected)
        assert "require macOS 26.0" in trace[0].reason


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


ROCM_VARIANTS = [
    parse_variant(f"torch215-cxx11-{rocm}-x86_64-linux") for rocm in ["rocm72", "rocm100", "rocm101", "rocm102"]
]


def _resolve_rocm(rocm_version: str):
    return _resolve_variant_for_system(
        variants=ROCM_VARIANTS,
        selected_backend=ROCm(Version(rocm_version)),
        cpu="x86_64",
        os="linux",
        torch_version=Version("2.15"),
        torch_cxx11_abi=True,
        tvm_ffi_version=None,
    )


def test_resolve_rocm_best_older_minor():
    # Newest minor that is not newer than the system, same major only.
    result, _ = _resolve_rocm("10.1")
    assert [v.variant_str for v in result] == [
        "torch215-cxx11-rocm101-x86_64-linux",
        "torch215-cxx11-rocm100-x86_64-linux",
    ]


def test_resolve_rocm_no_different_major():
    result, _ = _resolve_rocm("7.2")
    assert [v.variant_str for v in result] == ["torch215-cxx11-rocm72-x86_64-linux"]


def test_resolve_rocm_no_newer_minor():
    result, _ = _resolve_rocm("10.0")
    assert [v.variant_str for v in result] == ["torch215-cxx11-rocm100-x86_64-linux"]


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
