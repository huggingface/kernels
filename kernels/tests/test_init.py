import tempfile
from pathlib import Path
import argparse
import os

from kernels.cli.init import run_init, parse_kernel_name
from kernels.utils import KNOWN_BACKENDS


def e2e_init(backends: list[str]) -> None:
    kernel_name = "testuser/test-kernel"
    template_repo = "drbh/template"
    args = argparse.Namespace(
        kernel_name=parse_kernel_name(kernel_name),
        template_repo=template_repo,
        backends=backends,
        overwrite=False,
    )
    expected_dir_name = "test-kernel"
    expected_normalized_name = "test_kernel"
    expected_backend_dirs = {
        Path(f"{expected_normalized_name}_{backend}") for backend in args.backends
    }

    # Replacement logic
    # special case for "rocm" backend since it uses "cuda" source
    if "rocm" in args.backends:
        expected_backend_dirs.remove(Path(f"{expected_normalized_name}_rocm"))
        expected_backend_dirs.add(Path(f"{expected_normalized_name}_cuda"))
    if "all" in args.backends:
        expected_backend_dirs = {
            Path(f"{expected_normalized_name}_{backend}") for backend in KNOWN_BACKENDS
        }
        # special case for "rocm" backend since it uses "cuda" source
        expected_backend_dirs.remove(Path(f"{expected_normalized_name}_rocm"))
        expected_backend_dirs.add(Path(f"{expected_normalized_name}_cuda"))

    # TODO: npu is not yet supported in the template
    expected_backend_dirs.discard(Path(f"{expected_normalized_name}_npu"))

    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = Path.cwd()
        os.chdir(tmpdir)
        try:
            run_init(args)

            # make sure dir was created
            target_dir = Path(tmpdir) / expected_dir_name
            if not target_dir.exists():
                raise AssertionError(f"Target directory was not created: {target_dir}")

            # check that expected backend dirs were created
            for expected_backend_dir in expected_backend_dirs:
                if not target_dir.joinpath(expected_backend_dir).exists():
                    raise AssertionError(
                        f"Expected backend directory was not created: {expected_backend_dir}"
                    )

        finally:
            os.chdir(cwd)


def test_end_to_end_init() -> None:
    e2e_init(backends=["cuda", "rocm"])
    e2e_init(backends=["metal", "cpu"])
    e2e_init(backends=["all"])
