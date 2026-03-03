import pytest
from unittest.mock import MagicMock

from kernels.status import (
    Redirect,
    KernelStatus,
    resolve_status,
)


class TestKernelStatusFromToml:
    def test_simple_redirect(self):
        content = '''kind = "redirect"
destination = "kernels-community/new-kernel"'''
        result = KernelStatus.from_toml(content)
        assert isinstance(result, Redirect)
        assert result.destination == "kernels-community/new-kernel"
        assert result.revision == "main"

    def test_redirect_with_revision(self):
        content = '''kind = "redirect"
destination = "kernels-community/new-kernel"
revision = "v2"'''
        result = KernelStatus.from_toml(content)
        assert isinstance(result, Redirect)
        assert result.destination == "kernels-community/new-kernel"
        assert result.revision == "v2"

    def test_missing_kind_raises(self):
        with pytest.raises(ValueError, match="must contain a 'kind' field"):
            KernelStatus.from_toml('destination = "kernels-community/new-kernel"')

    def test_unknown_kind_raises(self):
        content = '''kind = "unknown"
destination = "kernels-community/new-kernel"'''
        with pytest.raises(ValueError, match="Unknown kernel status kind"):
            KernelStatus.from_toml(content)

    def test_missing_destination_raises(self):
        with pytest.raises(ValueError, match="must contain a 'destination'"):
            KernelStatus.from_toml('kind = "redirect"')

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="must contain a 'kind'"):
            KernelStatus.from_toml("")


class TestResolveStatus:
    def test_no_status(self):
        from huggingface_hub.utils import EntryNotFoundError

        mock_api = MagicMock()
        mock_api.hf_hub_download.side_effect = EntryNotFoundError("Not found")

        repo_id, revision = resolve_status(mock_api, "kernels-test/kernel", "main")
        assert repo_id == "kernels-test/kernel"
        assert revision == "main"

    def test_redirect(self, tmp_path):
        from huggingface_hub.utils import EntryNotFoundError

        status_file = tmp_path / "kernel-status.toml"
        status_file.write_text('kind = "redirect"\ndestination = "kernels-community/new-kernel"')

        def mock_download(repo_id, filename, revision):
            if repo_id == "kernels-test/old-kernel":
                return str(status_file)
            raise EntryNotFoundError("Not found")

        mock_api = MagicMock()
        mock_api.hf_hub_download.side_effect = mock_download

        repo_id, revision = resolve_status(mock_api, "kernels-test/old-kernel", "main")
        assert repo_id == "kernels-community/new-kernel"
        assert revision == "main"

    def test_redirect_with_revision(self, tmp_path):
        from huggingface_hub.utils import EntryNotFoundError

        status_file = tmp_path / "kernel-status.toml"
        status_file.write_text('kind = "redirect"\ndestination = "kernels-community/new-kernel"\nrevision = "v2"')

        def mock_download(repo_id, filename, revision):
            if repo_id == "kernels-test/old-kernel":
                return str(status_file)
            raise EntryNotFoundError("Not found")

        mock_api = MagicMock()
        mock_api.hf_hub_download.side_effect = mock_download

        repo_id, revision = resolve_status(mock_api, "kernels-test/old-kernel", "main")
        assert repo_id == "kernels-community/new-kernel"
        assert revision == "v2"
