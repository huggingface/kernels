"""Best-effort upgrade suggestion when no build variant matches the current revision."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import LocalEntryNotFoundError

from kernels._versions import _get_available_versions
from kernels.backends import CANN, CUDA, ROCm, XPU
from kernels.hf_hub import CACHE_DIR
from kernels.variants import (
    ArchVariant,
    NoarchVariant,
    Torch,
    TorchStableAbi,
    TvmFfi,
    Variant,
    get_variants,
    get_variants_local,
    resolve_variant,
)

_MAX_NEWER_SCANS = 16
_BYTECODE_IGNORE_PATTERNS = ["*.pyc", "**/__pycache__/**"]


def _parse_version_revision(revision: str) -> int | None:
    """Return N when revision is exactly ``v<N>`` with a digit suffix; else None."""
    if not revision.startswith("v") or len(revision) < 2:
        return None
    try:
        return int(revision[1:])
    except ValueError:
        return None


def _env_identity_from_variant(variant: Variant) -> str:
    """Short parenthetical env identity from an accepted variant (omit missing pieces)."""
    parts: list[str] = []

    if isinstance(variant, ArchVariant):
        framework = variant.framework
        if isinstance(framework, Torch):
            parts.append(f"torch {framework.version.major}.{framework.version.minor}")
        elif isinstance(framework, TorchStableAbi):
            parts.append(f"torch-stable-abi {framework.version.major}.{framework.version.minor}")
        elif isinstance(framework, TvmFfi):
            parts.append(f"tvm-ffi {framework.version.major}.{framework.version.minor}")

        backend = variant.arch.backend
        if isinstance(backend, (CUDA, ROCm, XPU, CANN)):
            parts.append(f"{backend.name} {backend.version.major}.{backend.version.minor}")
        else:
            parts.append(backend.name)
    elif isinstance(variant, NoarchVariant):
        parts.append(variant.arch.backend_name)

    return ", ".join(parts)


def _format_suggestion(repo_id: str, version: int, env_identity: str) -> str:
    identity = f" ({env_identity})" if env_identity else ""
    return (
        f"However, version v{version} of '{repo_id}' has a build compatible "
        f"with your system{identity}. Consider upgrading to that version."
    )


def _variants_for_revision(
    api: HfApi,
    repo_id: str,
    revision: str,
    *,
    local_files_only: bool,
) -> list[Variant]:
    if local_files_only:
        local_repo_path = Path(
            str(
                api.snapshot_download(
                    repo_id,
                    repo_type="kernel",
                    ignore_patterns=_BYTECODE_IGNORE_PATTERNS,
                    cache_dir=CACHE_DIR,
                    revision=revision,
                    local_files_only=True,
                )
            )
        )
        return get_variants_local(local_repo_path / "build")
    return get_variants(api, repo_id=repo_id, revision=revision)


def maybe_upgrade_hint(
    repo_id: str,
    revision: str,
    *,
    api: HfApi,
    backend: str | None = None,
    local_files_only: bool = False,
) -> str | None:
    """Return an upgrade suggestion string, or None if none / best-effort failed.

    Never raises into the caller's error path.
    """
    try:
        current = _parse_version_revision(revision)
        if current is None:
            return None

        versions = _get_available_versions(repo_id)
        newer = sorted((v for v in versions if v > current), reverse=True)

        for version in newer[:_MAX_NEWER_SCANS]:
            try:
                ref = versions[version]
                variants = _variants_for_revision(
                    api,
                    repo_id,
                    ref.name,
                    local_files_only=local_files_only,
                )
                variant, _ = resolve_variant(variants, backend)
                if variant is not None:
                    return _format_suggestion(
                        repo_id,
                        version,
                        _env_identity_from_variant(variant),
                    )
            except LocalEntryNotFoundError:
                continue
            except Exception:
                continue

        return None
    except Exception:
        return None
