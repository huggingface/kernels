from unittest.mock import MagicMock, mock_open, patch

import pytest

from kernels.cli.collate_readme import (
    collate_readme_from_versions,
    get_latest_version_readme,
)


def _make_versions(version_nums: list[int]) -> dict:
    versions = {}
    for v in version_nums:
        ref = MagicMock()
        ref.name = f"v{v}"
        ref.ref = f"refs/heads/v{v}"
        versions[v] = ref
    return versions


PATCH_VERSIONS = "kernels.cli.collate_readme._get_available_versions"
PATCH_API = "kernels.cli.collate_readme._get_hf_api"

SAMPLE_README = """\
---
license: apache-2.0
library_name: kernels
---

# my-kernel

This is the repository card of org/my-kernel.

## How to use

```python
from kernels import get_kernel
kernel_module = get_kernel("org/my-kernel")
```
"""


class TestGetLatestVersionReadme:
    @patch("builtins.open", mock_open(read_data=SAMPLE_README))
    @patch(PATCH_API)
    @patch(PATCH_VERSIONS)
    def test_fetches_latest_version(self, mock_versions, mock_api):
        mock_versions.return_value = _make_versions([1, 2, 3])
        api_instance = MagicMock()
        api_instance.hf_hub_download.return_value = "/tmp/README.md"
        mock_api.return_value = api_instance

        readme = get_latest_version_readme("org/my-kernel")

        api_instance.hf_hub_download.assert_called_once_with(
            repo_id="org/my-kernel",
            filename="README.md",
            revision="v3",
        )
        assert readme == SAMPLE_README

    @patch("builtins.open", mock_open(read_data=SAMPLE_README))
    @patch(PATCH_API)
    @patch(PATCH_VERSIONS)
    def test_picks_highest_version(self, mock_versions, mock_api):
        mock_versions.return_value = _make_versions([3, 1, 5, 2])
        api_instance = MagicMock()
        api_instance.hf_hub_download.return_value = "/tmp/README.md"
        mock_api.return_value = api_instance

        get_latest_version_readme("org/kernel")

        api_instance.hf_hub_download.assert_called_once_with(
            repo_id="org/kernel",
            filename="README.md",
            revision="v5",
        )

    @patch(PATCH_VERSIONS)
    def test_no_versions_raises(self, mock_versions):
        mock_versions.return_value = {}

        with pytest.raises(ValueError, match="No versions found"):
            get_latest_version_readme("org/kernel")


class TestCollateReadmeCLI:
    @patch("builtins.open", mock_open(read_data=SAMPLE_README))
    @patch(PATCH_API)
    @patch(PATCH_VERSIONS)
    def test_prints_to_stdout(self, mock_versions, mock_api, capsys):
        mock_versions.return_value = _make_versions([1, 2])
        api_instance = MagicMock()
        api_instance.hf_hub_download.return_value = "/tmp/README.md"
        mock_api.return_value = api_instance

        collate_readme_from_versions(repo_id="org/my-kernel")

        captured = capsys.readouterr()
        assert "# my-kernel" in captured.out

    @patch("builtins.open", mock_open(read_data=SAMPLE_README))
    @patch(PATCH_API)
    @patch(PATCH_VERSIONS)
    def test_push_to_hub(self, mock_versions, mock_api):
        mock_versions.return_value = _make_versions([1])
        api_instance = MagicMock()
        api_instance.hf_hub_download.return_value = "/tmp/README.md"
        mock_api.return_value = api_instance

        collate_readme_from_versions(repo_id="org/kernel", push_to_hub=True)

        api_instance.upload_file.assert_called_once()
        call_kwargs = api_instance.upload_file.call_args[1]
        assert call_kwargs["path_in_repo"] == "README.md"
        assert call_kwargs["repo_id"] == "org/kernel"
        assert call_kwargs["revision"] == "main"
