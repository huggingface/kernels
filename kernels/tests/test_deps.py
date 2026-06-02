from importlib.util import find_spec
from pathlib import Path

import pytest
from huggingface_hub import HfApi, snapshot_download

from kernels import get_kernel, get_local_kernel, install_kernel


@pytest.mark.cuda_only
@pytest.mark.parametrize("dependency", ["einops", "nvidia-cutlass-dsl"])
def test_python_deps(dependency):
    must_raise = find_spec(dependency.replace("-", "_")) is None
    if must_raise:
        with pytest.raises(
            ImportError,
            match=r"Kernel module `python_dep` requires Python dependency `(einops|nvidia-cutlass-dsl)`",
        ):
            get_kernel("kernels-test/python-dep", revision="main")
    else:
        get_kernel("kernels-test/python-dep", revision="main")


def test_illegal_dep():
    with pytest.raises(ValueError, match=r"Kernel module `python_invalid_dep` uses.*kepler-22b"):
        get_kernel("kernels-test/python-invalid-dep", revision="main")


def test_deps_validated_before_download(monkeypatch):
    """With `validate_dependencies=True`, deps are checked *before* the build
    variant is downloaded.

    `snapshot_download` (the full-variant download) is patched to fail, so the
    dependency error can only surface if validation runs first — guarding the
    early-bail ordering against regressions that move validation back after the
    download.
    """

    class _Downloaded(RuntimeError):
        pass

    def _fail(*_args, **_kwargs):
        raise _Downloaded("build variant was downloaded before dependency validation")

    monkeypatch.setattr(HfApi, "snapshot_download", _fail)

    with pytest.raises(ValueError, match=r"Kernel module `python_invalid_dep` uses.*kepler-22b"):
        install_kernel("kernels-test/python-invalid-dep", revision="main", validate_dependencies=True)


def test_install_kernel_skips_validation_by_default():
    """Validation is opt-in: with the default `validate_dependencies=False`,
    `install_kernel` downloads an invalid-dep kernel without raising."""
    variant_path = install_kernel("kernels-test/python-invalid-dep", revision="main")
    assert (variant_path / "metadata.json").exists()


def test_local_kernel_validates_deps(tmp_path):
    """`get_local_kernel` validates dependencies even though it never goes through
    `install_kernel`'s pre-download check, so removing the gate from
    `_import_from_path` must not drop validation for local kernels."""
    repo_path = snapshot_download(
        "kernels-test/python-invalid-dep",
        repo_type="kernel",
        revision="main",
        cache_dir=tmp_path,
    )
    with pytest.raises(ValueError, match=r"Kernel module `python_invalid_dep` uses.*kepler-22b"):
        get_local_kernel(Path(repo_path))
