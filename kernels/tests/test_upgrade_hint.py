"""Unit tests for upgrade-hint suggestion on no-variant errors (#744)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.hf_api import GitRefInfo

from kernels._upgrade_hint import (
    _MAX_NEWER_SCANS,
    _env_identity_from_variant,
    maybe_upgrade_hint,
)
from kernels.install import _resolve_local_variant_path, install_kernel
from kernels.variants import VariantRejected, parse_variant, variants_trace_str


def _ref(version: int) -> GitRefInfo:
    return GitRefInfo(name=f"v{version}", ref=f"v{version}", target_commit=f"c{version}")


COMPATIBLE = parse_variant("torch213-cxx11-cu128-x86_64-linux")


def _resolve_by_revision(calls: list[str]):
    """Side-effect factory: accept only when last get_variants revision is in accept_set."""

    def _make(accept: set[str]):
        def resolve_variant(variants, backend=None):
            revision = calls[-1] if calls else ""
            if revision in accept:
                return COMPATIBLE, []
            return None, []

        return resolve_variant

    return _make


# --- Helper unit tests (T1–T5) ---


def test_suggests_newest_compatible():
    """T1 / A1: newest compatible newer v<N> is suggested with env identity."""
    versions = {3: _ref(3), 4: _ref(4), 5: _ref(5)}
    calls: list[str] = []
    api = MagicMock()

    def get_variants(api_arg, *, repo_id, revision):
        calls.append(revision)
        return []

    with (
        patch("kernels._upgrade_hint._get_available_versions", return_value=versions),
        patch("kernels._upgrade_hint.get_variants", side_effect=get_variants),
        patch(
            "kernels._upgrade_hint.resolve_variant",
            side_effect=_resolve_by_revision(calls)({"v5", "v4"}),
        ),
    ):
        hint = maybe_upgrade_hint("kernels-community/activation", "v3", api=api)

    assert hint is not None
    assert "However, version v5 of 'kernels-community/activation'" in hint
    assert "torch 2.13, cuda 12.8" in hint
    assert "Consider upgrading to that version." in hint
    # Newest-first: stop at v5; never consult v4.
    assert calls == ["v5"]
    assert "v4" not in hint


def test_no_suggestion_when_none_compatible():
    """T2 / A2: newer versions exist but none compatible → no suggestion."""
    versions = {3: _ref(3), 4: _ref(4), 5: _ref(5)}
    calls: list[str] = []
    api = MagicMock()

    def get_variants(api_arg, *, repo_id, revision):
        calls.append(revision)
        return []

    with (
        patch("kernels._upgrade_hint._get_available_versions", return_value=versions),
        patch("kernels._upgrade_hint.get_variants", side_effect=get_variants),
        patch(
            "kernels._upgrade_hint.resolve_variant",
            side_effect=_resolve_by_revision(calls)(set()),
        ),
    ):
        hint = maybe_upgrade_hint("kernels-community/activation", "v3", api=api)

    assert hint is None
    assert calls == ["v5", "v4"]


def test_suggestion_path_swallows_errors():
    """T3 / A3: Hub/version failures must not escape; return None."""
    api = MagicMock()

    with patch(
        "kernels._upgrade_hint._get_available_versions",
        side_effect=RuntimeError("hub down"),
    ):
        assert maybe_upgrade_hint("repo/id", "v3", api=api) is None

    versions = {3: _ref(3), 4: _ref(4)}
    with (
        patch("kernels._upgrade_hint._get_available_versions", return_value=versions),
        patch(
            "kernels._upgrade_hint.get_variants",
            side_effect=RuntimeError("list_repo_tree failed"),
        ),
    ):
        assert maybe_upgrade_hint("repo/id", "v3", api=api) is None


def test_offline_empty_cache_no_suggestion():
    """T4 / A4: HF_HUB_OFFLINE with empty enumeration → no hang, no suggestion."""
    api = MagicMock()

    with (
        patch("kernels._versions.constants.HF_HUB_OFFLINE", True),
        patch("kernels._upgrade_hint._get_available_versions", return_value={}),
    ):
        hint = maybe_upgrade_hint("repo/id", "v3", api=api, local_files_only=True)

    assert hint is None


def test_scan_cap_skips_beyond_16():
    """T5 / Q3: only the first 16 newer versions (newer-first) are consulted."""
    current = 1
    # 17 newer versions: 2..18. Newer-first order: 18,17,...,3 then 2 is 17th.
    versions = {v: _ref(v) for v in range(current, 19)}
    calls: list[str] = []
    api = MagicMock()

    def get_variants(api_arg, *, repo_id, revision):
        calls.append(revision)
        return []

    # Only the 17th-scanned candidate (v2) would be compatible — must NOT be seen.
    with (
        patch("kernels._upgrade_hint._get_available_versions", return_value=versions),
        patch("kernels._upgrade_hint.get_variants", side_effect=get_variants),
        patch(
            "kernels._upgrade_hint.resolve_variant",
            side_effect=_resolve_by_revision(calls)({"v2"}),
        ),
    ):
        hint = maybe_upgrade_hint("repo/id", "v1", api=api)

    assert hint is None
    assert len(calls) == _MAX_NEWER_SCANS
    assert calls == [f"v{v}" for v in range(18, 18 - _MAX_NEWER_SCANS, -1)]
    assert "v2" not in calls


def test_non_version_revision_skips_suggestion():
    """Non-v<N> revisions must not claim an upgrade path."""
    api = MagicMock()
    with patch("kernels._upgrade_hint._get_available_versions") as get_versions:
        assert maybe_upgrade_hint("repo/id", "main", api=api) is None
        assert maybe_upgrade_hint("repo/id", "abc123def", api=api) is None
        get_versions.assert_not_called()


def test_env_identity_from_variant():
    variant = parse_variant("torch213-cxx11-cu128-x86_64-linux")
    assert _env_identity_from_variant(variant) == "torch 2.13, cuda 12.8"

    cpu = parse_variant("torch25-cpu-x86_64-linux")
    assert _env_identity_from_variant(cpu) == "torch 2.5, cpu"

    noarch = parse_variant("torch-cuda")
    assert _env_identity_from_variant(noarch) == "cuda"


# --- install.py wiring (error message shape) ---


def test_install_kernel_appends_suggestion():
    """T1 via install_kernel: FileNotFoundError includes However clause."""
    api = MagicMock()
    rejected = VariantRejected(
        variant=parse_variant("torch25-cxx11-cu118-x86_64-linux"),
        reason="Torch version (2.5) does not match environment Torch version (2.13)",
    )
    suggestion = (
        "However, version v5 of 'kernels-community/activation' has a build compatible "
        "with your system (torch 2.13, cuda 12.8). Consider upgrading to that version."
    )

    with (
        patch("kernels.install._get_hf_api", return_value=api),
        patch("kernels.install.resolve_status", return_value=("kernels-community/activation", "v3")),
        patch("kernels.install.get_variants", return_value=[rejected.variant]),
        patch("kernels.install.resolve_variant", return_value=(None, [rejected])),
        patch("kernels.install.maybe_upgrade_hint", return_value=suggestion),
    ):
        with pytest.raises(FileNotFoundError) as exc_info:
            install_kernel("kernels-community/activation", revision="v3")

    msg = str(exc_info.value)
    assert msg.startswith(
        "Cannot find a build variant for this system in kernels-community/activation (revision: v3):"
    )
    assert variants_trace_str([rejected]) in msg
    assert "However, version v5" in msg
    assert "torch 2.13, cuda 12.8" in msg


def test_install_kernel_no_false_upgrade():
    """T2 via install_kernel: no However when helper returns None."""
    api = MagicMock()
    rejected = VariantRejected(
        variant=parse_variant("torch25-cxx11-cu118-x86_64-linux"),
        reason="incompatible",
    )

    with (
        patch("kernels.install._get_hf_api", return_value=api),
        patch("kernels.install.resolve_status", return_value=("repo/id", "v3")),
        patch("kernels.install.get_variants", return_value=[rejected.variant]),
        patch("kernels.install.resolve_variant", return_value=(None, [rejected])),
        patch("kernels.install.maybe_upgrade_hint", return_value=None),
    ):
        with pytest.raises(FileNotFoundError) as exc_info:
            install_kernel("repo/id", revision="v3")

    msg = str(exc_info.value)
    assert "However" not in msg
    assert msg.startswith("Cannot find a build variant for this system in repo/id (revision: v3):")


def test_local_resolve_preserves_error_when_hint_absent():
    """T3 via local path: missing suggestion keeps original FileNotFoundError."""
    api = MagicMock()
    api.snapshot_download.return_value = "/tmp/fake-kernel-cache"
    rejected = VariantRejected(
        variant=parse_variant("torch25-cxx11-cu118-x86_64-linux"),
        reason="incompatible",
    )

    with (
        patch("kernels.install.get_variants_local", return_value=[rejected.variant]),
        patch("kernels.install.resolve_variant", return_value=(None, [rejected])),
        patch("kernels.install.maybe_upgrade_hint", return_value=None),
    ):
        with pytest.raises(FileNotFoundError) as exc_info:
            _resolve_local_variant_path(api, "repo/id", revision="v3")

    msg = str(exc_info.value)
    assert "However" not in msg
    assert "Cannot find a build variant for this system in repo/id (revision: v3):" in msg
