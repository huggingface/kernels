import pytest

from kernels_data import Backend, Build, KernelDependency, KernelVersion


def _write_build_toml(path, backends):
    backends_toml = ", ".join(f'"{b}"' for b in backends)
    path.write_text(
        f"""\
[general]
name = "my-kernel"
version = 1
edition = 5
license = "Apache-2.0"
backends = [{backends_toml}]
kernel-depends = [{{ repo-id = "kernels-community/activation", version = 1 }}]

[general.cuda]
kernel-depends = [{{ repo-id = "kernels-community/cuda-helper", version = 2 }}]

[torch-noarch]
"""
    )
    return path


def test_build_open_and_general_backends(tmp_path):
    _write_build_toml(tmp_path / "build.toml", backends=["cpu", "cuda"])

    build = Build.open(tmp_path)
    assert isinstance(build, Build)
    assert build.general.backends == [Backend.CPU, Backend.CUDA]


def test_build_open_missing_file(tmp_path):
    with pytest.raises(ValueError):
        Build.open(tmp_path)


def test_build_all_kernel_depends(tmp_path):
    _write_build_toml(tmp_path / "build.toml", backends=["cpu", "cuda"])

    build = Build.open(tmp_path)
    assert build.all_kernel_depends(Backend.CUDA) == [
        KernelDependency(
            repo_id="kernels-community/activation", version=KernelVersion.Version(1)
        ),
        KernelDependency(
            repo_id="kernels-community/cuda-helper", version=KernelVersion.Version(2)
        ),
    ]
    assert build.all_kernel_depends(Backend.CPU) == [
        KernelDependency(
            repo_id="kernels-community/activation", version=KernelVersion.Version(1)
        )
    ]


def test_build_all_kernel_depends_empty(tmp_path):
    (tmp_path / "build.toml").write_text(
        """\
[general]
name = "my-kernel"
version = 1
edition = 5
license = "Apache-2.0"
backends = ["cpu"]

[torch-noarch]
"""
    )

    build = Build.open(tmp_path)
    assert build.all_kernel_depends(Backend.CPU) == []
