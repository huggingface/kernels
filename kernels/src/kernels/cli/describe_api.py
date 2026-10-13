import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import constants
from huggingface_hub.errors import RemoteEntryNotFoundError

from kernels._versions import _get_available_versions, resolve_version_spec_as_ref
from kernels.hf_hub import _get_cache_dir, _get_hf_api
from kernels.variants import Variant, get_variants, get_variants_local


def print_kernel_api(
    kernel: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    json_output: bool = False,
):
    """Describe saved public symbols without downloading or importing kernel code."""
    try:
        if revision is not None and version is not None:
            raise ValueError("Only one of `revision` or `version` can be specified")

        path = Path(kernel).expanduser()
        if path.is_dir():
            if revision is not None or version is not None:
                raise ValueError("`revision` and `version` cannot be used with a local path")
            symbols_path, variant = _local_symbols(path)
            source = {"path": str(path), "variant": variant}
        else:
            if path.exists() or path.is_absolute() or kernel.startswith(("./", "../", "~")):
                raise ValueError(f"Not a local kernel directory: {kernel}")
            symbols_path, revision, variant = _hub_symbols(kernel, revision=revision, version=version)
            source = {"repo_id": kernel, "revision": revision, "variant": variant}

        info = {**_read_symbols(symbols_path), **source}
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    if json_output:
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        _print_human(info)


def _find_variant_symbols(
    variants: list[Variant], get_path: Callable[[str], Path | None], *, location: str
) -> tuple[Path, str]:
    variant = min((v.variant_str for v in variants), default=None)
    if variant is not None:
        location = f"{location} (variant: {variant})"
        symbols_path = get_path(f"{variant}/symbols.json")
        if symbols_path is not None:
            return symbols_path, variant
    raise ValueError(
        f"No symbols.json found in {location}. "
        "Rebuild the kernel with a recent kernel-builder and the get_kernel check enabled."
    )


def _local_symbols(path: Path) -> tuple[Path, str]:
    if (path / "symbols.json").is_file():
        return path / "symbols.json", path.name

    build_path = path / "build" if (path / "build").is_dir() else path

    def get_path(filename: str) -> Path | None:
        symbols_path = build_path / filename
        return symbols_path if symbols_path.is_file() else None

    return _find_variant_symbols(get_variants_local(build_path), get_path, location=str(path))


def _hub_symbols(repo_id: str, *, revision: str | None, version: int | None) -> tuple[Path, str, str]:
    if version is not None:
        revision = resolve_version_spec_as_ref(repo_id, version, local_files_only=constants.HF_HUB_OFFLINE).name
    elif revision is None:
        versions = _get_available_versions(repo_id, local_files_only=constants.HF_HUB_OFFLINE)
        version = max(versions, default=None)
        revision = versions[version].name if version is not None else "main"

    location = f"{repo_id} (revision: {revision})"
    if version is not None:
        location = f"{repo_id} (version: {version}, revision: {revision})"

    api = _get_hf_api()
    try:
        variants = get_variants(api, repo_id=repo_id, revision=revision)
    except RemoteEntryNotFoundError:
        variants = []

    def get_path(filename: str) -> Path | None:
        try:
            symbols_path = api.hf_hub_download(
                repo_id,
                repo_type="kernel",
                filename=f"build/{filename}",
                revision=revision,
                cache_dir=_get_cache_dir(),
                local_files_only=constants.HF_HUB_OFFLINE,
            )
        except RemoteEntryNotFoundError:
            return None
        return Path(symbols_path)

    symbols_path, variant = _find_variant_symbols(variants, get_path, location=location)
    return symbols_path, revision, variant


def _read_symbols(path: Path) -> dict[str, Any]:
    try:
        symbols = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(f"Invalid symbols.json at {path}: {error}") from error
    if not isinstance(symbols, dict) or symbols.get("schema_version") != 1:
        raise ValueError(f"Unsupported symbols.json schema at {path}; expected schema_version 1")

    for group in ("functions", "layers"):
        exports = symbols.get(group)
        if not isinstance(exports, list) or any(
            not isinstance(symbol, dict) or not isinstance(symbol.get("name"), str) for symbol in exports
        ):
            raise ValueError(f"Invalid symbols.json at {path}: {group} must be a list of named symbols")
    for layer in symbols["layers"]:
        attributes = layer.get("attributes")
        if not isinstance(attributes, dict) or any(
            value is not None and not isinstance(value, bool) for value in attributes.values()
        ):
            raise ValueError(f"Invalid symbols.json at {path}: layer attributes must be booleans or null")
    return symbols


def _print_human(info: dict[str, Any]) -> None:
    if "repo_id" in info:
        print(f"Repository: {info['repo_id']}")
        print(f"Revision: {info['revision']}")
    else:
        print(f"Path: {info['path']}")
    print(f"Variant: {info['variant']}")

    print("\nFunctions:")
    for function in info["functions"]:
        print(f"  {function['name']}")
    if not info["functions"]:
        print("  (none)")

    print("\nLayers:")
    for layer in info["layers"]:
        attributes = layer["attributes"]
        flags = "  ".join(
            f"{name}={attributes[name] if attributes.get(name) is not None else 'unknown'}"
            for name in ("has_backward", "can_torch_compile")
        )
        print(f"  {layer['name']}  {flags}")
    if not info["layers"]:
        print("  (none)")
