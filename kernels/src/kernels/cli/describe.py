"""Describe kernel exports by reading Python source, without executing it."""

import ast
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from huggingface_hub import constants
from huggingface_hub.errors import HfHubHTTPError, RemoteEntryNotFoundError

from kernels._rust import KernelName, Metadata
from kernels._versions import _get_available_versions
from kernels.compat import tomllib
from kernels.hf_hub import _get_cache_dir, _get_hf_api
from kernels.variants import get_variants, get_variants_local

_Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def print_kernel_description(
    kernel: str,
    *,
    version: int | Literal["latest"] | None = None,
    revision: str | None = None,
    json_output: bool = False,
):
    """Print the functions and layers explicitly exported by a kernel."""
    try:
        if version is not None and revision is not None:
            raise ValueError("Only one of revision or version can be specified")

        path = Path(kernel)
        if path.is_dir():
            if version is not None or revision is not None:
                raise ValueError("revision and version cannot be used with a local path")
            source = _local_source(path)
            info: dict[str, Any] = {"path": str(path)}
        else:
            if revision is None:
                versions = _get_available_versions(kernel, local_files_only=constants.HF_HUB_OFFLINE)
                if isinstance(version, int):
                    if version not in versions:
                        raise ValueError(
                            f"Version {version} not found, available versions: {', '.join(map(str, sorted(versions)))}"
                        )
                    revision = versions[version].name
                else:
                    revision = versions[max(versions)].name if versions else "main"
            source = _hub_source(kernel, revision)
            info = {"repo_id": kernel, "revision": revision}

        functions = source.exports("__init__.py", (ast.FunctionDef, ast.AsyncFunctionDef))
        layers_file = source.module_file("layers")
        layers = source.exports(layers_file, (ast.ClassDef,)) if layers_file else []
    except (ValueError, OSError, SyntaxError, HfHubHTTPError) as error:
        print(f"Cannot describe {kernel}: {error}", file=sys.stderr)
        sys.exit(1)

    info["functions"] = [name for name, _, _ in functions]
    info["layers"] = []
    for name, node, _ in layers:
        flags = _assignments(node.body)
        layer: dict[str, Any] = {"name": name}
        for flag in ("has_backward", "can_torch_compile"):
            value = flags.get(flag)
            layer[flag] = value.value if isinstance(value, ast.Constant) and isinstance(value.value, bool) else None
        info["layers"].append(layer)

    if json_output:
        print(json.dumps(info, indent=2))
    else:
        _print_human(info)


def _print_human(info: dict):
    if "repo_id" in info:
        print(f"Repository: {info['repo_id']}")
        print(f"Revision: {info['revision']}")
    else:
        print(f"Path: {info['path']}")

    print("\nFunctions:")
    for name in info["functions"]:
        print(f"  {name}")
    if not info["functions"]:
        print("  No functions declared in __all__.")

    print("\nLayers:")
    for layer in info["layers"]:
        capabilities = []
        for flag in ("has_backward", "can_torch_compile"):
            rendered = "unknown" if layer[flag] is None else str(layer[flag])
            capabilities.append(f"{flag}={rendered}")
        print(f"  {layer['name']}  {'  '.join(capabilities)}")
    if not info["layers"]:
        print("  No layers declared in __all__.")


def _local_source(path: Path) -> "_Source":
    # Accept a package/variant directory as well as a built repository.
    if (path / "__init__.py").is_file():
        return _Source(path)
    variants = get_variants_local(path / "build")
    if variants:
        path = path / "build" / min(v.variant_str for v in variants)
    elif (path / "build.toml").is_file():
        with (path / "build.toml").open("rb") as config_file:
            config = tomllib.load(config_file)
        name = config.get("general", {}).get("name")
        if not isinstance(name, str):
            raise ValueError("Missing general.name in build.toml")
        name = KernelName(name).python_name
        for directory in ("torch-ext", "tvm-ffi-ext"):
            package = path / directory / name
            if (package / "__init__.py").is_file():
                return _Source(package)
        raise ValueError(f"No Python kernel package found in {path}")
    return _source_from_variant(_Source(path))


def _hub_source(repo_id: str, revision: str) -> "_Source":
    api = _get_hf_api()
    variants = get_variants(api, repo_id=repo_id, revision=revision)
    if not variants:
        raise ValueError(f"No build variants found (revision: {revision})")
    # API source can be inspected on any host; do not resolve compatibility or
    # download a snapshot (which would also fetch the compiled kernel).
    variant = min(v.variant_str for v in variants)
    return _source_from_variant(_Source(Path("build") / variant, repo_id=repo_id, revision=revision))


def _source_from_variant(source: "_Source") -> "_Source":
    if source.read("__init__.py") is not None:
        return source
    metadata_path = source.file("metadata.json")
    if metadata_path is not None:
        name = Metadata.read_from_file(metadata_path).name.python_name
        source = _Source(source.root / name, repo_id=source.repo_id, revision=source.revision)
        if source.read("__init__.py") is not None:
            return source
    raise ValueError("No Python kernel package found")


def _assignments(body: list[ast.stmt]) -> dict[str, ast.expr]:
    values = {}
    for node in body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            values[node.target.id] = node.value
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if node.target.id in values:
                values[node.target.id] = ast.BinOp(values[node.target.id], node.op, node.value)
    return values


def _export_names(tree: ast.Module, filename: str) -> list[str]:
    value = _assignments(tree.body).get("__all__")
    if value is None:
        return []

    def strings(node: ast.expr) -> list[str]:
        if isinstance(node, (ast.List, ast.Tuple)):
            names = [
                item.value for item in node.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)
            ]
            if len(names) == len(node.elts):
                return names
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return strings(node.left) + strings(node.right)
        raise ValueError(f"Cannot statically read __all__ in {filename}; expected a literal list or tuple of names")

    return list(dict.fromkeys(strings(value)))


class _Source:
    """Read only API source files and relative re-exports, never import them."""

    def __init__(self, root: Path, *, repo_id: str | None = None, revision: str | None = None):
        self.root = root
        self.repo_id = repo_id
        self.revision = revision
        self._modules: dict[str, ast.Module | None] = {}

    def file(self, filename: str) -> Path | None:
        path = self.root / filename
        if self.repo_id is None:
            return path if path.is_file() else None
        try:
            return Path(
                _get_hf_api().hf_hub_download(
                    self.repo_id,
                    repo_type="kernel",
                    filename=path.as_posix(),
                    revision=self.revision,
                    cache_dir=_get_cache_dir(),
                    local_files_only=constants.HF_HUB_OFFLINE,
                )
            )
        except RemoteEntryNotFoundError:
            return None

    def read(self, filename: str) -> ast.Module | None:
        if filename not in self._modules:
            path = self.file(filename)
            self._modules[filename] = ast.parse(path.read_bytes(), filename=filename) if path is not None else None
        return self._modules[filename]

    def module_file(self, module: str) -> str | None:
        for filename in (f"{module}/__init__.py", f"{module}.py"):
            if self.read(filename) is not None:
                return filename
        return None

    def resolve(
        self, filename: str, name: str, seen: frozenset[tuple[str, str]] = frozenset()
    ) -> tuple[_Definition, str] | None:
        if (filename, name) in seen or "_private_for_testing" in PurePosixPath(filename).parts:
            return None
        seen = seen | {(filename, name)}
        tree = self.read(filename)
        if tree is None:
            return None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                return node, filename
            if isinstance(node, ast.ImportFrom) and node.level and node.module:
                for alias in node.names:
                    if (alias.asname or alias.name) != name:
                        continue
                    parts = list(PurePosixPath(filename).parent.parts)
                    if node.level > len(parts) + 1:
                        return None
                    parts = parts[: len(parts) - node.level + 1] + node.module.split(".")
                    if "_private_for_testing" in parts:
                        return None
                    target = self.module_file("/".join(parts))
                    return self.resolve(target, alias.name, seen) if target else None
        value = _assignments(tree.body).get(name)
        if isinstance(value, ast.Name):
            return self.resolve(filename, value.id, seen)
        return None

    def exports(self, filename: str, kinds: tuple[type, ...]) -> list[tuple[str, _Definition, str]]:
        tree = self.read(filename)
        if tree is None:
            raise ValueError(f"Missing Python source: {filename}")
        exports = []
        for name in _export_names(tree, filename):
            if name.startswith("_") or name == "layers":
                continue
            resolved = self.resolve(filename, name)
            if resolved is None:
                print(f"Cannot statically resolve export {name!r} in {filename}", file=sys.stderr)
            elif isinstance(resolved[0], kinds):
                exports.append((name, *resolved))
        return exports
