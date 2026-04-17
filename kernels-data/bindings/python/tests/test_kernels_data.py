import json

import pytest

from kernels_data import Backend, KernelName, Metadata, Version


def _write_metadata(path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields))
    return path


def test_version_parse_and_normalize():
    assert str(Version.from_str("12.8.0")) == "12.8"
    assert str(Version.from_str("1")) == "1"
    assert str(Version.from_str("1.2.3")) == "1.2.3"


def test_version_ordering_and_hash():
    v1 = Version.from_str("1.2")
    v2 = Version.from_str("1.2.0")
    v3 = Version.from_str("1.3")
    assert v1 == v2
    assert v1 < v3
    assert hash(v1) == hash(v2)
    assert {v1, v2, v3} == {v1, v3}


def test_version_invalid():
    with pytest.raises(ValueError):
        Version.from_str("abc")
    with pytest.raises(ValueError):
        Version.from_str("")


def test_kernel_name_valid():
    n = KernelName("my-kernel")
    assert str(n) == "my-kernel"
    assert n.python_name == "my_kernel"


def test_kernel_name_hash_and_eq():
    assert KernelName("flash-attention") == KernelName("flash-attention")
    assert {KernelName("a1"), KernelName("a1")} == {KernelName("a1")}


def test_kernel_name_invalid():
    with pytest.raises(ValueError):
        KernelName("My-Kernel")
    with pytest.raises(ValueError):
        KernelName("1kernel")
    with pytest.raises(ValueError):
        KernelName("-kernel")


def test_backend_from_str_and_repr():
    assert Backend.from_str("cuda") == Backend.CUDA
    assert Backend.from_str("CUDA") == Backend.CUDA
    assert str(Backend.CUDA) == "cuda"
    assert repr(Backend.CUDA) == "Backend.CUDA"


def test_backend_hash():
    d = {Backend.CUDA: 1, Backend.CPU: 2}
    assert d[Backend.CUDA] == 1


def test_backend_unknown():
    with pytest.raises(ValueError):
        Backend.from_str("tpu")


def test_backend_all_variants_and_casing():
    assert str(Backend.Metal) == "metal"
    assert repr(Backend.Metal) == "Backend.Metal"
    assert str(Backend.Neuron) == "neuron"
    assert repr(Backend.Neuron) == "Backend.Neuron"
    assert str(Backend.ROCm) == "rocm"
    assert repr(Backend.ROCm) == "Backend.ROCm"
    assert repr(Backend.XPU) == "Backend.XPU"
    assert repr(Backend.CANN) == "Backend.CANN"
    assert Backend.from_str("cann") == Backend.CANN
    assert Backend.from_str("ROCM") == Backend.ROCm
    assert Backend.from_str("metal") == Backend.Metal


def test_metadata_load_full(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "license": "Apache-2.0",
                "upstream": "https://github.com/example/kernel",
                "python-depends": ["torch"],
                "backend": {"type": "cuda"},
            }
        )
    )
    m = Metadata.load(path)
    assert m.version == 1
    assert m.license == "Apache-2.0"
    assert m.upstream == "https://github.com/example/kernel"
    assert m.python_depends == ["torch"]
    assert m.backend == Backend.CUDA


def test_metadata_load_minimal(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"python-depends": [], "backend": {"type": "cpu"}}))
    m = Metadata.load(path)
    assert m.version is None
    assert m.license is None
    assert m.upstream is None
    assert m.python_depends == []
    assert m.backend == Backend.CPU


def test_metadata_load_cann(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"python-depends": [], "backend": {"type": "cann"}}))
    assert Metadata.load(path).backend == Backend.CANN


def test_metadata_load_unknown_field_accepted(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "python-depends": [],
                "backend": {"type": "cpu"},
                "surprise": "not allowed",
            }
        )
    )
    Metadata.load(path)


def test_metadata_load_malformed(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text("{not json")
    with pytest.raises(ValueError):
        Metadata.load(path)


def test_metadata_load(tmp_path):
    path = _write_metadata(
        tmp_path / "variant" / "metadata.json",
        **{"python-depends": ["torch"], "backend": {"type": "cuda"}},
    )
    m = Metadata.load(path)
    assert m.backend == Backend.CUDA


def test_metadata_load_missing_file(tmp_path):
    with pytest.raises(ValueError):
        Metadata.load(tmp_path / "does-not-exist.json")
