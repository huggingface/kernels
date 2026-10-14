import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, constants
from huggingface_hub.errors import RemoteEntryNotFoundError

from kernels._rust import KernelDependency
from kernels._versions import _get_available_versions, revision_or_version
from kernels.hf_hub import _get_hf_api
from kernels.resolver import HubCacheResolver, HubResolver, LocalKernel, RemoteKernel, resolve_local_kernel


def print_kernel_api(
    kernel: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    json_output: bool = False,
    trust_remote_code: bool = False,
):
    """Describe saved public symbols without importing kernel code."""
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
            symbols_path, revision, variant = _hub_symbols(
                kernel, revision=revision, version=version, trust_remote_code=trust_remote_code
            )
            source = {"repo_id": kernel, "revision": revision, "variant": variant}

        info = {**_read_symbols(symbols_path), **source}
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    if json_output:
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        _print_human(info)


def _symbols_path(kernel: LocalKernel | RemoteKernel, *, api: HfApi, location: str) -> Path:
    try:
        symbols_path = kernel.get_file("symbols.json", api=api)
    except RemoteEntryNotFoundError:
        pass
    else:
        if symbols_path.is_file():
            return symbols_path
    raise ValueError(
        f"No symbols.json found in {location} (variant: {kernel.variant_str}). "
        "Rebuild the kernel with a recent kernel-builder and the get_kernel check enabled."
    )


def _local_symbols(path: Path) -> tuple[Path, str]:
    kernel = resolve_local_kernel(path, backend=None)
    return _symbols_path(kernel, api=_get_hf_api(), location=str(path)), kernel.variant_str


def _hub_symbols(
    repo_id: str, *, revision: str | None, version: int | None, trust_remote_code: bool
) -> tuple[Path, str, str]:
    if revision is None and version is None:
        versions = _get_available_versions(repo_id, local_files_only=constants.HF_HUB_OFFLINE)
        if not versions:
            raise ValueError(f"No numbered versions found for {repo_id}. Specify --revision explicitly.")
        version = max(versions)

    dependency = KernelDependency(repo_id=repo_id, version=revision_or_version(revision=revision, version=version))
    revision = f"v{version}" if version is not None else revision
    assert revision is not None

    location = f"{repo_id} (revision: {revision})"
    if version is not None:
        location = f"{repo_id} (version: {version}, revision: {revision})"

    api = _get_hf_api()
    resolver = (
        HubCacheResolver(trust_remote_code=trust_remote_code)
        if constants.HF_HUB_OFFLINE
        else HubResolver(trust_remote_code=trust_remote_code)
    )
    try:
        kernel = resolver.resolve(api=api, backend=None, kernel=dependency)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot resolve {location}: {error}") from error

    return _symbols_path(kernel, api=api, location=location), revision, kernel.variant_str


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
