import tempfile
from pathlib import Path
from dataclasses import dataclass

import pytest

from kernels.cli import create_and_upload_card


@dataclass
class CardArgs:
    kernel_dir: str
    card_path: str
    description: str | None = None
    repo_id: str | None = None
    create_pr: bool = False


@pytest.fixture
def mock_kernel_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        kernel_dir = Path(tmpdir)

        build_toml = kernel_dir / "build.toml"
        build_toml.write_text(
            """[general]
name = "test_kernel"
backends = ["cuda", "metal"]
license = "apache-2.0"
version = 1

[general.hub]
repo-id = "test-org/test-kernel"

[kernel._test]
backend = "cuda"
cuda-capabilities = ["8.0", "8.9"]
"""
        )

        torch_ext_dir = kernel_dir / "torch-ext" / "test_kernel"
        torch_ext_dir.mkdir(parents=True)

        init_file = torch_ext_dir / "__init__.py"
        init_file.write_text(
            """from .core import func1, func2

__all__ = ["func1", "func2", "func3"]
"""
        )

        core_file = torch_ext_dir / "core.py"
        core_file.write_text(
            """def func1():
    pass

def func2():
    pass

def func3():
    pass
"""
        )

        yield kernel_dir


@pytest.fixture
def mock_kernel_dir_with_benchmark(mock_kernel_dir):
    benchmarks_dir = mock_kernel_dir / "benchmarks"
    benchmarks_dir.mkdir()

    benchmark_file = benchmarks_dir / "benchmark.py"
    benchmark_file.write_text(
        """import time

def benchmark():
    # Simple benchmark
    start = time.time()
    # ... benchmark code ...
    end = time.time()
    return end - start
"""
    )

    return mock_kernel_dir


@pytest.fixture
def mock_kernel_dir_minimal():
    with tempfile.TemporaryDirectory() as tmpdir:
        kernel_dir = Path(tmpdir)

        build_toml = kernel_dir / "build.toml"
        build_toml.write_text(
            """[general]
name = "minimal_kernel"
backends = ["cuda"]
"""
        )

        yield kernel_dir


def test_create_and_upload_card_basic(mock_kernel_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        card_path = Path(tmpdir) / "README.md"

        args = CardArgs(
            kernel_dir=str(mock_kernel_dir),
            card_path=str(card_path),
            description="This is a test kernel for testing purposes.",
        )

        create_and_upload_card(args)

        assert card_path.exists()

        card_content = card_path.read_text()

        assert "---" in card_content
        assert "This is a test kernel for testing purposes." in card_content


def test_create_and_upload_card_updates_usage(mock_kernel_dir):
    """Test that usage code snippet is properly generated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        card_path = Path(tmpdir) / "README.md"

        args = CardArgs(
            kernel_dir=str(mock_kernel_dir),
            card_path=str(card_path),
        )

        create_and_upload_card(args)

        card_content = card_path.read_text()

        assert "## How to use" in card_content
        assert "from kernels import get_kernel" in card_content
        assert "func1" in card_content
        assert "TODO: add an example code snippet" not in card_content


def test_create_and_upload_card_updates_available_functions(mock_kernel_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        card_path = Path(tmpdir) / "README.md"

        args = CardArgs(
            kernel_dir=str(mock_kernel_dir),
            card_path=str(card_path),
        )

        create_and_upload_card(args)

        card_content = card_path.read_text()

        assert "## Available functions" in card_content
        assert "- `func1`" in card_content
        assert "- `func2`" in card_content
        assert "- `func3`" in card_content
        assert (
            "[TODO: add the functions available through this kernel]"
            not in card_content
        )


def test_create_and_upload_card_updates_backends(mock_kernel_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        card_path = Path(tmpdir) / "README.md"

        args = CardArgs(
            kernel_dir=str(mock_kernel_dir),
            card_path=str(card_path),
        )

        create_and_upload_card(args)

        card_content = card_path.read_text()

        assert "## Supported backends" in card_content
        assert "- cuda" in card_content
        assert "- metal" in card_content
        assert "[TODO: add the backends this kernel supports]" not in card_content


def test_create_and_upload_card_updates_cuda_capabilities(mock_kernel_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        card_path = Path(tmpdir) / "README.md"

        args = CardArgs(
            kernel_dir=str(mock_kernel_dir),
            card_path=str(card_path),
        )

        create_and_upload_card(args)

        card_content = card_path.read_text()

        assert "## CUDA Capabilities" in card_content
        assert "- 8.0" in card_content or "- 8.9" in card_content


def test_create_and_upload_card_updates_license(mock_kernel_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        card_path = Path(tmpdir) / "README.md"

        args = CardArgs(
            kernel_dir=str(mock_kernel_dir),
            card_path=str(card_path),
        )

        create_and_upload_card(args)

        card_content = card_path.read_text()

        assert "license: apache-2.0" in card_content


def test_create_and_upload_card_with_benchmark(mock_kernel_dir_with_benchmark):
    with tempfile.TemporaryDirectory() as tmpdir:
        card_path = Path(tmpdir) / "README.md"

        args = CardArgs(
            kernel_dir=str(mock_kernel_dir_with_benchmark),
            card_path=str(card_path),
        )

        create_and_upload_card(args)

        card_content = card_path.read_text()

        assert "## Benchmarks" in card_content
        assert "Benchmarking script is available for this kernel" in card_content
        assert "kernels benchmark" in card_content


def test_create_and_upload_card_minimal_structure(mock_kernel_dir_minimal):
    with tempfile.TemporaryDirectory() as tmpdir:
        card_path = Path(tmpdir) / "README.md"

        args = CardArgs(
            kernel_dir=str(mock_kernel_dir_minimal),
            card_path=str(card_path),
        )

        create_and_upload_card(args)

        assert card_path.exists()

        card_content = card_path.read_text()

        assert "---" in card_content
        assert "## How to use" in card_content
        assert "## Available functions" in card_content
        assert "## Supported backends" in card_content


def test_create_and_upload_card_custom_description(mock_kernel_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        card_path = Path(tmpdir) / "README.md"

        custom_desc = "My custom kernel description with special features."

        args = CardArgs(
            kernel_dir=str(mock_kernel_dir),
            card_path=str(card_path),
            description=custom_desc,
        )

        create_and_upload_card(args)

        card_content = card_path.read_text()

        assert custom_desc in card_content
