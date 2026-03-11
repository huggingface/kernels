import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from kernels.cli import fill_kernel_card, initialize_card
from kernels.cli.kernel_card_utils import KERNEL_CARD_TEMPLATE_PATH

SYSTEM_CARD_PATH = "CARD.md"


@dataclass
class CardArgs:
    kernel_dir: str
    repo_id: str | None = None
    description: str | None = None


@pytest.fixture
def mock_kernel_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        kernel_dir = Path(tmpdir)

        (kernel_dir / "build.toml").write_text("""[general]
name = "test_kernel"
backends = ["cuda", "metal"]
license = "apache-2.0"
version = 1

[general.hub]
repo-id = "my-org/my-kernel"

[kernel._test]
backend = "cuda"
cuda-capabilities = ["8.0", "8.9"]
""")

        torch_ext_dir = kernel_dir / "torch-ext" / "test_kernel"
        torch_ext_dir.mkdir(parents=True)
        (torch_ext_dir / "__init__.py").write_text(
            'from .core import func1, func2\n\n__all__ = ["func1", "func2"]\n'
        )
        (torch_ext_dir / "core.py").write_text(
            "def func1():\n    pass\n\ndef func2():\n    pass\n"
        )

        yield kernel_dir


@pytest.fixture
def initialized_kernel_dir(mock_kernel_dir):
    shutil.copy(KERNEL_CARD_TEMPLATE_PATH, mock_kernel_dir / SYSTEM_CARD_PATH)
    return mock_kernel_dir


def test_initialize_card_creates_file(mock_kernel_dir):
    args = CardArgs(kernel_dir=str(mock_kernel_dir))
    initialize_card(args)
    card_path = mock_kernel_dir / SYSTEM_CARD_PATH
    assert card_path.exists()

    content = card_path.read_text()
    assert "{{ kernel_description }}" in content
    assert "{{ supported_backends }}" in content


def test_fill_kernel_card_backends(initialized_kernel_dir):
    args = CardArgs(kernel_dir=str(initialized_kernel_dir))
    fill_kernel_card(args)
    content = (initialized_kernel_dir / SYSTEM_CARD_PATH).read_text()
    assert "- cuda" in content
    assert "- metal" in content


def test_fill_kernel_card_cuda_capabilities(initialized_kernel_dir):
    args = CardArgs(kernel_dir=str(initialized_kernel_dir))
    fill_kernel_card(args)
    content = (initialized_kernel_dir / SYSTEM_CARD_PATH).read_text()
    assert "## CUDA Capabilities" in content
    assert "- 8.0" in content or "- 8.9" in content


def test_fill_kernel_card_available_funcs(initialized_kernel_dir):
    args = CardArgs(kernel_dir=str(initialized_kernel_dir))
    fill_kernel_card(args)
    content = (initialized_kernel_dir / SYSTEM_CARD_PATH).read_text()
    assert "- `func1`" in content
    assert "- `func2`" in content


def test_fill_kernel_card_usage_repo_id(initialized_kernel_dir):
    args = CardArgs(kernel_dir=str(initialized_kernel_dir))
    fill_kernel_card(args)
    content = (initialized_kernel_dir / SYSTEM_CARD_PATH).read_text()
    assert 'get_kernel("my-org/my-kernel")' in content


def test_fill_kernel_card_usage_with_build_toml_repo_id(initialized_kernel_dir):
    args = CardArgs(kernel_dir=str(initialized_kernel_dir))
    fill_kernel_card(args)
    content = (initialized_kernel_dir / SYSTEM_CARD_PATH).read_text()
    assert 'get_kernel("my-org/my-kernel")' in content


def test_fill_kernel_card_placeholder_repo_id(mock_kernel_dir):
    (mock_kernel_dir / "build.toml").write_text("""[general]
name = "test_kernel"
backends = ["cuda"]
license = "apache-2.0"
version = 1

[kernel._test]
backend = "cuda"
cuda-capabilities = ["8.0"]
""")
    shutil.copy(KERNEL_CARD_TEMPLATE_PATH, mock_kernel_dir / SYSTEM_CARD_PATH)

    args = CardArgs(kernel_dir=str(mock_kernel_dir))
    fill_kernel_card(args)
    content = (mock_kernel_dir / SYSTEM_CARD_PATH).read_text()
    assert 'get_kernel("{repo_id}")' in content
    assert "{repo_id}" in content


def test_fill_kernel_card_license(initialized_kernel_dir):
    args = CardArgs(kernel_dir=str(initialized_kernel_dir))
    fill_kernel_card(args)
    content = (initialized_kernel_dir / SYSTEM_CARD_PATH).read_text()
    assert "license: apache-2.0" in content


def test_fill_kernel_card_benchmark(initialized_kernel_dir):
    benchmarks_dir = initialized_kernel_dir / "benchmarks"
    benchmarks_dir.mkdir()
    (benchmarks_dir / "benchmark.py").write_text("def benchmark(): pass\n")
    args = CardArgs(kernel_dir=str(initialized_kernel_dir))
    fill_kernel_card(args)
    content = (initialized_kernel_dir / SYSTEM_CARD_PATH).read_text()
    assert "## Benchmarks" in content
    assert "Benchmarking script is available" in content


def test_fill_kernel_card_preserves_existing_content(mock_kernel_dir):
    existing_description = "A hand-written description of this kernel."
    existing_notes = "Custom notes that should not be overwritten."
    existing_source = "https://github.com/example/kernel-source"

    card_path = mock_kernel_dir / SYSTEM_CARD_PATH
    card_path.write_text(
        "---\n"
        "{{ card_data }}\n"
        "---\n\n"
        f"{existing_description}\n\n"
        "{{ kernel_description }}\n\n"
        "## How to use\n\n"
        "{{ usage_example }}\n\n"
        "## Available functions\n\n"
        "{{ available_functions }}\n\n"
        "## Supported backends\n\n"
        "{{ supported_backends }}\n\n"
        f"## Source code\n\n{existing_source}\n\n"
        "{{ source_code }}\n\n"
        f"## Notes\n\n{existing_notes}\n"
    )

    args = CardArgs(kernel_dir=str(mock_kernel_dir))
    fill_kernel_card(args)

    content = card_path.read_text()

    # Dynamic content
    assert "- `func1`" in content
    assert "- `func2`" in content
    assert "- cuda" in content
    assert "- metal" in content
    assert 'get_kernel("my-org/my-kernel")' in content

    # Static user content
    assert existing_description in content
    assert existing_notes in content
    assert existing_source in content


def test_fill_kernel_card_upstream_source(mock_kernel_dir):
    build_toml = mock_kernel_dir / "build.toml"
    upstream = "huggingface/upstream-kernel"
    build_toml.write_text(
        f'upstream = "{upstream}"\n'
        "\n"
        "[general]\n"
        'name = "test_kernel"\n'
        'backends = ["cuda", "metal"]\n'
        'license = "apache-2.0"\n'
        "version = 1\n"
        "\n"
        "[kernel._test]\n"
        'backend = "cuda"\n'
        'cuda-capabilities = ["8.0", "8.9"]\n'
    )

    shutil.copy(KERNEL_CARD_TEMPLATE_PATH, mock_kernel_dir / SYSTEM_CARD_PATH)

    args = CardArgs(kernel_dir=str(mock_kernel_dir))
    fill_kernel_card(args)
    content = (mock_kernel_dir / SYSTEM_CARD_PATH).read_text()
    assert f"{upstream}" in content


def test_fill_kernel_card_preserves_user_notes(initialized_kernel_dir):
    card_path = initialized_kernel_dir / SYSTEM_CARD_PATH
    user_text = "Custom kernel notes."
    existing_content = card_path.read_text()
    card_path.write_text(existing_content.replace("{{ notes }}", user_text))

    args = CardArgs(kernel_dir=str(initialized_kernel_dir))
    fill_kernel_card(args)
    content = card_path.read_text()
    assert user_text in content
