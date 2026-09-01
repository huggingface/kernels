import json

import pytest

from kernels_data import (
    Backend,
    KernelDependency,
    KernelName,
    KernelVersion,
    Metadata,
    Version,
)


def _write_metadata(path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields))
    return path


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
        Backend.from_str("dsp")


def test_backend_all_variants_and_casing():
    assert str(Backend.Metal) == "metal"
    assert repr(Backend.Metal) == "Backend.Metal"
    assert str(Backend.Neuron) == "neuron"
    assert repr(Backend.Neuron) == "Backend.Neuron"
    assert str(Backend.ROCm) == "rocm"
    assert repr(Backend.ROCm) == "Backend.ROCm"
    assert repr(Backend.XPU) == "Backend.XPU"
    assert repr(Backend.CANN) == "Backend.CANN"
    assert str(Backend.TPU) == "tpu"
    assert repr(Backend.TPU) == "Backend.TPU"
    assert Backend.from_str("cann") == Backend.CANN
    assert Backend.from_str("ROCM") == Backend.ROCm
    assert Backend.from_str("TPU") == Backend.TPU
    assert Backend.from_str("metal") == Backend.Metal


def test_metadata_load_full(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "id": "_my_kernel_8a3be8f",
                "version": 1,
                "name": "my-kernel",
                "license": "Apache-2.0",
                "kernels-minver": "0.17.0",
                "upstream": "https://github.com/example/kernel",
                "source": "https://github.com/example/kernel-builder",
                "python-depends": ["torch"],
                "backend": {"type": "cuda", "archs": ["9.0", "10.0"]},
            }
        )
    )
    m = Metadata.read_from_file(path)
    assert m.id == "_my_kernel_8a3be8f"
    assert m.name == KernelName("my-kernel")
    assert m.version == 1
    assert m.kernels_minver == Version.from_str("0.17.0")
    assert m.license == "Apache-2.0"
    assert m.upstream == "https://github.com/example/kernel"
    assert m.source == "https://github.com/example/kernel-builder"
    assert m.python_depends == ["torch"]
    assert m.backend.backend_type == Backend.CUDA
    assert m.backend.archs == ["9.0", "10.0"]


def test_metadata_load_minimal(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "id": "_my_kernel_8a3be8f",
                "version": 1,
                "name": "my-kernel",
                "license": "Apache-2.0",
                "python-depends": [],
                "backend": {"type": "cpu"},
            }
        )
    )
    m = Metadata.read_from_file(path)
    assert m.version == 1
    assert m.kernels_minver is None
    assert m.license == "Apache-2.0"
    assert m.upstream is None
    assert m.source is None
    assert m.python_depends == []
    assert m.backend.backend_type == Backend.CPU


def test_metadata_load_cann(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "id": "_my_kernel_8a3be8f",
                "version": 1,
                "name": "my-kernel",
                "license": "Apache-2.0",
                "python-depends": [],
                "backend": {"type": "cann"},
            }
        )
    )
    assert Metadata.read_from_file(path).backend.backend_type == Backend.CANN


def test_metadata_load_unknown_field_accepted(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "id": "_my_kernel_8a3be8f",
                "version": 1,
                "name": "my-kernel",
                "license": "Apache-2.0",
                "python-depends": [],
                "backend": {"type": "cpu"},
                "surprise": "not allowed",
            }
        )
    )
    Metadata.read_from_file(path)


def test_metadata_load_malformed(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text("{not json")
    with pytest.raises(ValueError):
        Metadata.read_from_file(path)


def test_metadata_load(tmp_path):
    path = _write_metadata(
        tmp_path / "variant" / "metadata.json",
        **{
            "id": "_my_kernel_8a3be8f",
            "version": 1,
            "name": "my-kernel",
            "license": "Apache-2.0",
            "python-depends": ["torch"],
            "backend": {"type": "cuda"},
        },
    )
    m = Metadata.read_from_file(path)
    assert m.backend.backend_type == Backend.CUDA


def test_metadata_load_missing_file(tmp_path):
    with pytest.raises(OSError):
        Metadata.read_from_file(tmp_path / "does-not-exist.json")


def test_kernel_version_eq_and_hash():
    assert KernelVersion.Version(1) == KernelVersion.Version(1)
    assert KernelVersion.Revision("abc") == KernelVersion.Revision("abc")
    assert KernelVersion.Version(1) != KernelVersion.Version(2)
    assert KernelVersion.Version(1) != KernelVersion.Revision("1")
    assert hash(KernelVersion.Version(1)) == hash(KernelVersion.Version(1))
    assert hash(KernelVersion.Revision("abc")) == hash(KernelVersion.Revision("abc"))
    assert {KernelVersion.Version(1), KernelVersion.Version(1)} == {
        KernelVersion.Version(1)
    }


def test_kernel_dependency_eq_and_hash():
    a = KernelDependency(repo_id="foo/bar", version=KernelVersion.Version(1))
    b = KernelDependency(repo_id="foo/bar", version=KernelVersion.Version(1))
    c = KernelDependency(repo_id="foo/bar", version=KernelVersion.Revision("deadbeef"))
    d = KernelDependency(repo_id="other/repo", version=KernelVersion.Version(1))

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != d

    # Distinct instances with equal content collapse in sets/dicts.
    assert {a, b, c, d} == {a, c, d}

    cache: dict[KernelDependency, int] = {a: 1}
    assert cache[b] == 1


def test_kernel_dependency_is_immutable():
    dep = KernelDependency(repo_id="foo/bar", version=KernelVersion.Version(1))
    with pytest.raises(AttributeError):
        dep.repo_id = "other/repo"
    with pytest.raises(AttributeError):
        dep.version = KernelVersion.Version(2)
