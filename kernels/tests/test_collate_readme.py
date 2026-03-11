from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from kernels.cli.collate_readme import (
    collate_readme_from_versions,
    generate_main_readme,
)


def _make_versions(version_nums: list[int]) -> dict:
    versions = {}
    for v in version_nums:
        ref = MagicMock()
        ref.name = f"v{v}"
        ref.ref = f"refs/heads/v{v}"
        versions[v] = ref
    return versions


@dataclass
class CollateArgs:
    repo_id: str
    push_to_hub: bool = False


PATCH_VERSIONS = "kernels.cli.collate_readme._get_available_versions"
PATCH_API = "kernels.cli.collate_readme._get_hf_api"


class TestGenerateMainReadme:
    @patch(PATCH_VERSIONS)
    def test_basic_output(self, mock_versions):
        mock_versions.return_value = _make_versions([1, 2])
        readme = generate_main_readme("kernels-community/my-kernel")

        assert "# my-kernel" in readme
        assert "| 1 | [v1]" in readme
        assert "| 2 | [v2]" in readme
        assert "kernels-community/my-kernel" in readme

    @patch(PATCH_VERSIONS)
    def test_has_frontmatter(self, mock_versions):
        mock_versions.return_value = _make_versions([1])
        readme = generate_main_readme("org/kernel")

        assert readme.startswith("---\n")
        assert "tags:" in readme
        assert "- kernels" in readme
        assert "library_name: kernels" in readme

    @patch(PATCH_VERSIONS)
    def test_versions_sorted(self, mock_versions):
        mock_versions.return_value = _make_versions([3, 1, 2])
        readme = generate_main_readme("org/kernel")

        v1_pos = readme.index("| 1 |")
        v2_pos = readme.index("| 2 |")
        v3_pos = readme.index("| 3 |")
        assert v1_pos < v2_pos < v3_pos

    @patch(PATCH_VERSIONS)
    def test_hub_urls(self, mock_versions):
        mock_versions.return_value = _make_versions([1])
        readme = generate_main_readme("kernels-community/activation")

        assert "https://huggingface.co/kernels-community/activation/tree/v1" in readme

    @patch(PATCH_VERSIONS)
    def test_quick_start_section(self, mock_versions):
        mock_versions.return_value = _make_versions([1])
        readme = generate_main_readme("org/kernel")

        assert "## Quick start" in readme
        assert 'get_kernel("org/kernel"' in readme

    @patch(PATCH_VERSIONS)
    def test_no_versions_raises(self, mock_versions):
        mock_versions.return_value = {}

        with pytest.raises(ValueError, match="No versions found"):
            generate_main_readme("org/kernel")


class TestCollateReadmeCLI:
    @patch(PATCH_VERSIONS)
    def test_prints_to_stdout(self, mock_versions, capsys):
        mock_versions.return_value = _make_versions([1, 2])

        collate_readme_from_versions(repo_id="org/kernel")

        captured = capsys.readouterr()
        assert "# kernel" in captured.out
        assert "| 1 |" in captured.out

    @patch(PATCH_API)
    @patch(PATCH_VERSIONS)
    def test_push_to_hub(self, mock_versions, mock_api):
        mock_versions.return_value = _make_versions([1])
        api_instance = MagicMock()
        mock_api.return_value = api_instance

        collate_readme_from_versions(repo_id="org/kernel", push_to_hub=True)

        api_instance.upload_file.assert_called_once()
        call_kwargs = api_instance.upload_file.call_args[1]
        assert call_kwargs["path_in_repo"] == "README.md"
        assert call_kwargs["repo_id"] == "org/kernel"
        assert call_kwargs["revision"] == "main"
