"""Tests for offline kernel loading (issue #553).

When HF_HUB_OFFLINE=1 (or local_files_only=True), install_kernel() must
discover available build variants from the local cache instead of calling
api.list_repo_tree(), which fails with a network error in offline mode.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kernels.utils import install_kernel
from kernels.variants import Variant


def _make_fake_variant() -> Variant:
    """Return a minimal Variant object for mocking resolve_variant."""
    v = MagicMock(spec=Variant)
    v.variant_str = "torch26-cuda12-linux-x86_64"
    return v


class TestOfflineInstallKernel:
    """install_kernel() must use local cache when HF_HUB_OFFLINE=1."""

    def test_local_files_only_does_not_call_list_repo_tree(self, tmp_path):
        """With local_files_only=True, list_repo_tree() must never be called."""
        fake_variant = _make_fake_variant()
        repo_path = tmp_path / "snapshot"
        repo_path.mkdir()

        mock_api = MagicMock()
        mock_api.snapshot_download.return_value = str(repo_path)

        with (
            patch("kernels.utils._get_hf_api", return_value=mock_api),
            patch("kernels.utils.get_variants_local", return_value=[fake_variant]),
            patch("kernels.utils.resolve_variant", return_value=fake_variant),
            patch("kernels.utils._find_kernel_in_repo_path") as mock_find,
        ):
            mock_find.return_value = repo_path / "build" / fake_variant.variant_str
            install_kernel(
                "kernels-community/fake-kernel",
                revision="abc123",
                local_files_only=True,
            )

        mock_api.list_repo_tree.assert_not_called()

    def test_hf_hub_offline_env_does_not_call_list_repo_tree(self, tmp_path, monkeypatch):
        """With HF_HUB_OFFLINE=1 (via constants), list_repo_tree() must never
        be called and get_variants_local() must be used instead."""
        fake_variant = _make_fake_variant()
        repo_path = tmp_path / "snapshot"
        repo_path.mkdir()

        mock_api = MagicMock()
        mock_api.snapshot_download.return_value = str(repo_path)

        with (
            patch("kernels.utils._get_hf_api", return_value=mock_api),
            patch("kernels.utils.constants") as mock_constants,
            patch("kernels.utils.get_variants_local", return_value=[fake_variant]),
            patch("kernels.utils.resolve_variant", return_value=fake_variant),
            patch("kernels.utils._find_kernel_in_repo_path") as mock_find,
        ):
            mock_constants.HF_HUB_OFFLINE = True
            mock_constants.HF_HUB_DISABLE_TELEMETRY = False
            mock_find.return_value = repo_path / "build" / fake_variant.variant_str
            install_kernel(
                "kernels-community/fake-kernel",
                revision="abc123",
            )

        mock_api.list_repo_tree.assert_not_called()

    def test_local_files_only_passes_flag_to_snapshot_download(self, tmp_path):
        """snapshot_download() must be called with local_files_only=True in
        offline mode so it reads from cache without any network request."""
        fake_variant = _make_fake_variant()
        repo_path = tmp_path / "snapshot"
        repo_path.mkdir()

        mock_api = MagicMock()
        mock_api.snapshot_download.return_value = str(repo_path)

        with (
            patch("kernels.utils._get_hf_api", return_value=mock_api),
            patch("kernels.utils.get_variants_local", return_value=[fake_variant]),
            patch("kernels.utils.resolve_variant", return_value=fake_variant),
            patch("kernels.utils._find_kernel_in_repo_path") as mock_find,
        ):
            mock_find.return_value = repo_path / "build" / fake_variant.variant_str
            install_kernel(
                "kernels-community/fake-kernel",
                revision="abc123",
                local_files_only=True,
            )

        call_kwargs = mock_api.snapshot_download.call_args.kwargs
        assert call_kwargs.get("local_files_only") is True

    def test_local_files_only_uses_get_variants_local(self, tmp_path):
        """get_variants_local() must be called (not get_variants) in offline mode."""
        fake_variant = _make_fake_variant()
        repo_path = tmp_path / "snapshot"
        repo_path.mkdir()

        mock_api = MagicMock()
        mock_api.snapshot_download.return_value = str(repo_path)

        with (
            patch("kernels.utils._get_hf_api", return_value=mock_api),
            patch("kernels.utils.get_variants") as mock_get_variants,
            patch("kernels.utils.get_variants_local", return_value=[fake_variant]) as mock_local,
            patch("kernels.utils.resolve_variant", return_value=fake_variant),
            patch("kernels.utils._find_kernel_in_repo_path") as mock_find,
        ):
            mock_find.return_value = repo_path / "build" / fake_variant.variant_str
            install_kernel(
                "kernels-community/fake-kernel",
                revision="abc123",
                local_files_only=True,
            )

        mock_get_variants.assert_not_called()
        mock_local.assert_called_once_with(repo_path / "build")

    def test_missing_cache_raises_file_not_found(self, tmp_path):
        """When the kernel is not in the local cache, a FileNotFoundError
        with a helpful message must be raised."""
        from huggingface_hub.errors import LocalEntryNotFoundError

        mock_api = MagicMock()
        mock_api.snapshot_download.side_effect = LocalEntryNotFoundError("not cached")

        with patch("kernels.utils._get_hf_api", return_value=mock_api):
            with pytest.raises(FileNotFoundError, match="not available in the local cache"):
                install_kernel(
                    "kernels-community/not-cached",
                    revision="abc123",
                    local_files_only=True,
                )

    def test_online_path_still_calls_get_variants(self, tmp_path):
        """The online code path must not regress: get_variants() must still be
        called when local_files_only=False and HF_HUB_OFFLINE=False."""
        fake_variant = _make_fake_variant()
        repo_path = tmp_path / "snapshot"
        repo_path.mkdir()

        mock_api = MagicMock()
        mock_api.snapshot_download.return_value = str(repo_path)

        with (
            patch("kernels.utils._get_hf_api", return_value=mock_api),
            patch("kernels.utils.constants") as mock_constants,
            patch("kernels.utils.resolve_status", return_value=("kernels-community/fake-kernel", "abc123")),
            patch("kernels.utils.get_variants", return_value=[fake_variant]) as mock_get_variants,
            patch("kernels.utils.resolve_variant", return_value=fake_variant),
            patch("kernels.utils._find_kernel_in_repo_path") as mock_find,
        ):
            mock_constants.HF_HUB_OFFLINE = False
            mock_constants.HF_HUB_DISABLE_TELEMETRY = False
            mock_find.return_value = repo_path / "build" / fake_variant.variant_str
            install_kernel(
                "kernels-community/fake-kernel",
                revision="abc123",
                local_files_only=False,
            )

        mock_get_variants.assert_called_once()
