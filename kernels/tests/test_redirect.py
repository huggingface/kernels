import pytest
from unittest.mock import MagicMock

from kernels.redirect import (
    RedirectInfo,
    parse_redirect_file,
    resolve_redirect,
)


class TestParseRedirectFile:
    def test_simple_redirect(self):
        content = 'destination = "kernels-community/new-kernel"'
        result = parse_redirect_file(content)
        assert result.destination == "kernels-community/new-kernel"
        assert result.message == ""

    def test_redirect_with_message(self):
        content = '''destination = "kernels-community/new-kernel"
message = "This kernel has been moved to a new location."'''
        result = parse_redirect_file(content)
        assert result.destination == "kernels-community/new-kernel"
        assert "moved to a new location" in result.message

    def test_redirect_with_revision(self):
        content = 'destination = "kernels-community/new-kernel@v2"'
        result = parse_redirect_file(content)
        assert result.destination == "kernels-community/new-kernel@v2"

    def test_missing_destination_raises(self):
        with pytest.raises(ValueError, match="must contain a 'destination'"):
            parse_redirect_file('message = "no destination"')

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="must contain a 'destination'"):
            parse_redirect_file("")


class TestResolveRedirect:
    def test_no_redirect(self):
        from huggingface_hub.utils import EntryNotFoundError

        mock_api = MagicMock()
        mock_api.hf_hub_download.side_effect = EntryNotFoundError("Not found")

        repo_id, revision = resolve_redirect(mock_api, "kernels-test/kernel", "main")
        assert repo_id == "kernels-test/kernel"
        assert revision == "main"

    def test_single_redirect(self, tmp_path):
        from huggingface_hub.utils import EntryNotFoundError

        redirect_file = tmp_path / "REDIRECT.toml"
        redirect_file.write_text('destination = "kernels-community/new-kernel"\nmessage = "Kernel moved"')

        def mock_download(repo_id, filename, revision):
            if repo_id == "kernels-test/old-kernel":
                return str(redirect_file)
            raise EntryNotFoundError("Not found")

        mock_api = MagicMock()
        mock_api.hf_hub_download.side_effect = mock_download

        repo_id, revision = resolve_redirect(mock_api, "kernels-test/old-kernel", "main")
        assert repo_id == "kernels-community/new-kernel"
        assert revision == "main"

    def test_redirect_with_revision(self, tmp_path):
        from huggingface_hub.utils import EntryNotFoundError

        redirect_file = tmp_path / "REDIRECT.toml"
        redirect_file.write_text('destination = "kernels-community/new-kernel@v2"')

        def mock_download(repo_id, filename, revision):
            if repo_id == "kernels-test/old-kernel":
                return str(redirect_file)
            raise EntryNotFoundError("Not found")

        mock_api = MagicMock()
        mock_api.hf_hub_download.side_effect = mock_download

        repo_id, revision = resolve_redirect(mock_api, "kernels-test/old-kernel", "main")
        assert repo_id == "kernels-community/new-kernel"
        assert revision == "v2"
