from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from huggingface_hub import HfApi, constants

from kernels._rust import KernelDependency, Metadata, Oid
from kernels._versions import resolve_kernel_version
from kernels.backends import _select_backend
from kernels.hf_hub import _check_trust_remote_code, _get_cache_dir
from kernels.resolver import LocalKernel, RemoteKernel, Resolver
from kernels.variants import (
    get_variants,
    get_variants_local,
    parse_variant,
    resolve_variant,
    variants_trace_str,
)

logger = logging.getLogger(__name__)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GITHUB_KERNEL_SOURCE = "https://github.com/huggingface/kernels-community"
_JIT_CACHE_VERSION = "1"
_IGNORE_SOURCE_PARTS = frozenset({"__pycache__", ".git", "build", "_cmake_build"})


class JITBuildError(RuntimeError):
    """Raised when a kernel cannot be compiled locally."""


@dataclass(frozen=True)
class JITOptions:
    """Options for compiling a kernel when no compatible prebuilt variant exists.

    The source must be reproducible. For Hub kernels this is normally the
    source repository and commit recorded in the build metadata. ``source``
    can override the recorded source for kernels whose Hub metadata predates
    source recording.
    """

    source: str | Path | None = None
    source_subdir: str | None = None
    revision: str | None = None
    cache_dir: Path | None = None


@dataclass(frozen=True)
class _SourceSpec:
    source: str | Path
    subdir: str | None
    commit: str | None


@dataclass
class JITResolver:
    """Resolver that builds a CPU kernel when no compatible build exists."""

    options: JITOptions = field(default_factory=JITOptions)
    trust_remote_code: bool | list[str] = False

    def resolve(self, *, api: HfApi, backend: str | None, kernel: KernelDependency) -> LocalKernel:
        selected_backend = _select_backend(backend)
        if selected_backend.name != "cpu":
            raise JITBuildError(
                "The JIT fallback currently supports CPU kernels only; "
                f"the selected backend is `{selected_backend.name}`."
            )

        _check_trust_remote_code(
            repo_id=kernel.repo_id,
            local_files_only=False,
            trust_remote_code=self.trust_remote_code,
        )

        revision = resolve_kernel_version(
            kernel.repo_id,
            kernel.version,
            local_files_only=False,
        )
        candidates = _source_candidates(
            api=api,
            repo_id=kernel.repo_id,
            revision=revision,
            options=self.options,
        )
        if not candidates:
            raise JITBuildError(
                f"No JIT source is available for `{kernel.repo_id}`. "
                "Pass `jit=JITOptions(source=...)` with a pinned source tree "
                "or a GitHub repository URL."
            )

        errors: list[str] = []
        for candidate in candidates:
            try:
                return _build_local_kernel(
                    repo_id=kernel.repo_id,
                    revision=revision,
                    source=candidate,
                    options=self.options,
                )
            except Exception as exc:
                message = f"{candidate.source}@{candidate.commit or 'local'}: {exc}"
                errors.append(message)
                logger.warning("JIT build candidate failed: %s", message)

        raise JITBuildError(
            f"Could not compile a CPU kernel for `{kernel.repo_id}` from source. Tried:\n  - " + "\n  - ".join(errors)
        )


@dataclass
class FallbackResolver:
    """Use ``fallback`` only when the primary resolver has no compatible build."""

    primary: Resolver
    fallback: Resolver

    def resolve(
        self, *, api: HfApi, backend: str | None, kernel: KernelDependency
    ) -> LocalKernel | RemoteKernel | None:
        try:
            return self.primary.resolve(api=api, backend=backend, kernel=kernel)
        except FileNotFoundError:
            return self.fallback.resolve(api=api, backend=backend, kernel=kernel)


def _source_candidates(
    *,
    api: HfApi,
    repo_id: str,
    revision: Oid,
    options: JITOptions,
) -> list[_SourceSpec]:
    """Return source candidates in preferred order.

    The Hub metadata for a kernel can record the source repository. Older
    kernels do not have that field, so kernels in the ``kernels-community``
    organization fall back to the corresponding directory in the
    ``huggingface/kernels-community`` monorepo.
    """

    if options.source is not None:
        source_url_commit = _commit_from_source_url(options.source)
        commit = _resolve_source_commit(options.revision or source_url_commit, None)
        if commit is None and not _is_local_source(options.source):
            commit = _metadata_commit(
                api=api,
                repo_id=repo_id,
                revision=revision,
                source=options.source,
            )
        source, subdir = _normalize_source(options.source, options.source_subdir)
        return [_SourceSpec(source=source, subdir=subdir, commit=commit)]

    variants = get_variants(api, repo_id=repo_id, revision=str(revision))
    variants = sorted(variants, key=_variant_sort_key, reverse=True)

    candidates: list[_SourceSpec] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for variant in variants:
        try:
            metadata_path = Path(
                api.hf_hub_download(
                    repo_id,
                    repo_type="kernel",
                    filename=f"build/{variant.variant_str}/metadata.json",
                    cache_dir=_get_cache_dir(),
                    revision=str(revision),
                    local_files_only=False,
                )
            )
            metadata = Metadata.read_from_file(metadata_path)
        except Exception:
            continue

        source = metadata.source
        subdir = options.source_subdir
        if source is None and repo_id.startswith("kernels-community/"):
            source = _GITHUB_KERNEL_SOURCE
            subdir = subdir or str(metadata.name)
        if source is None:
            continue

        try:
            commit = _resolve_source_commit(
                options.revision or _commit_from_source_url(source),
                metadata.provenance.kernel.commit
                if metadata.provenance is not None and metadata.provenance.kernel is not None
                else None,
            )
            source, inferred_subdir = _normalize_source(source, subdir)
        except JITBuildError as exc:
            logger.warning(
                "Ignoring JIT source metadata for %s (%s): %s",
                repo_id,
                variant.variant_str,
                exc,
            )
            continue
        key = (str(source), inferred_subdir, commit)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(_SourceSpec(source=source, subdir=inferred_subdir, commit=commit))

    return candidates


def _metadata_commit(
    *,
    api: HfApi,
    repo_id: str,
    revision: Oid,
    source: str | Path,
) -> str | None:
    """Find the source commit for an explicitly supplied source."""

    if _is_local_source(source):
        return None

    variants = sorted(
        get_variants(api, repo_id=repo_id, revision=str(revision)),
        key=_variant_sort_key,
        reverse=True,
    )
    for variant in variants:
        try:
            metadata_path = Path(
                api.hf_hub_download(
                    repo_id,
                    repo_type="kernel",
                    filename=f"build/{variant.variant_str}/metadata.json",
                    cache_dir=_get_cache_dir(),
                    revision=str(revision),
                    local_files_only=False,
                )
            )
            metadata = Metadata.read_from_file(metadata_path)
        except Exception:
            continue
        if metadata.provenance is not None and metadata.provenance.kernel is not None:
            return str(metadata.provenance.kernel.commit)
    return None


def _variant_sort_key(variant) -> tuple[int, ...]:
    """Prefer newer Torch variants when choosing source metadata."""

    framework = getattr(variant, "framework", None)
    version = getattr(framework, "version", None)
    if version is None:
        return (0,)
    try:
        return tuple(int(part) for part in str(version).split("."))
    except ValueError:
        return (0,)


def _resolve_source_commit(explicit: str | None, recorded: str | None) -> str | None:
    if explicit is not None:
        if not _COMMIT_RE.fullmatch(explicit):
            raise JITBuildError(f"JIT source revisions must be full 40-character Git commit SHAs; got {explicit!r}.")
        return explicit
    if recorded is None:
        return None
    recorded = str(recorded)
    if not _COMMIT_RE.fullmatch(recorded):
        raise JITBuildError(f"Recorded source revision is not a full Git commit: {recorded!r}")
    return recorded


def _commit_from_source_url(source: str | Path) -> str | None:
    if _is_local_source(source):
        return None
    parts = [part for part in urllib.parse.urlsplit(str(source)).path.split("/") if part]
    if "tree" not in parts[2:]:
        return None
    tree_index = parts.index("tree", 2)
    if len(parts) <= tree_index + 1:
        return None
    candidate = parts[tree_index + 1]
    return candidate if _COMMIT_RE.fullmatch(candidate) else None


def _is_local_source(source: str | Path) -> bool:
    if isinstance(source, Path):
        return True
    parsed = urllib.parse.urlsplit(str(source))
    return parsed.scheme == "" and Path(str(source)).expanduser().exists()


def _normalize_source(source: str | Path, subdir: str | None) -> tuple[str | Path, str | None]:
    if isinstance(source, Path) or _is_local_source(source):
        path = Path(source).expanduser().resolve()
        if not path.is_dir():
            raise JITBuildError(f"JIT source directory does not exist: {path}")
        return path, subdir

    value = str(source)
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise JITBuildError(f"JIT source must be a local directory or an http(s) Git repository URL; got {value!r}.")
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise JITBuildError(f"JIT source URLs currently must point at github.com; got {parsed.netloc!r}.")

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        raise JITBuildError(f"Invalid GitHub source URL: {value!r}")
    repo = f"https://github.com/{path_parts[0]}/{path_parts[1].removesuffix('.git')}"

    if "tree" in path_parts[2:]:
        tree_index = path_parts.index("tree", 2)
        if len(path_parts) > tree_index + 2 and subdir is None:
            subdir = "/".join(path_parts[tree_index + 2 :])
    return repo, subdir


def _build_local_kernel(
    *,
    repo_id: str,
    revision: Oid,
    source: _SourceSpec,
    options: JITOptions,
) -> LocalKernel:
    _reject_unsupported_free_threaded_python()
    cache_root = _jit_cache_root(options.cache_dir)
    key = _cache_key(repo_id=repo_id, revision=revision, source=source)
    final_root = cache_root / key
    lock_path = cache_root / f"{key}.lock"

    with _exclusive_lock(lock_path):
        variant = _load_cached_variant(final_root)
        if variant is None:
            build_parent = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache_root))
            try:
                source_root = _materialize_source(source, build_parent)
                _run_build(source_root, source.commit)
                built_variant = _find_built_variant(source_root)
                final_root.mkdir(parents=True, exist_ok=True)
                final_variant = final_root / "build" / built_variant.name
                final_variant.parent.mkdir(parents=True, exist_ok=True)
                if final_variant.exists():
                    shutil.rmtree(final_variant)
                os.replace(built_variant, final_variant)
                variant = final_variant
            finally:
                shutil.rmtree(build_parent, ignore_errors=True)

    metadata = Metadata.read_from_file(variant / "metadata.json")
    origin = None
    try:
        origin = RemoteKernel(
            repo_id=repo_id,
            revision=revision,
            metadata=metadata,
            variant=parse_variant(variant.name),
        )
    except ValueError:
        # A locally generated variant name is still loadable; it just cannot be
        # represented as a remote build variant.
        pass
    return LocalKernel(variant_path=variant, metadata=metadata, origin=origin)


def _jit_cache_root(cache_dir: Path | None) -> Path:
    override = os.environ.get("KERNELS_JIT_CACHE")
    if cache_dir is not None:
        root = Path(cache_dir).expanduser()
    elif override:
        root = Path(override).expanduser()
    else:
        root = Path(_get_cache_dir() or constants.HF_HUB_CACHE) / "jit"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_key(*, repo_id: str, revision: Oid, source: _SourceSpec) -> str:
    import torch

    payload = {
        "version": _JIT_CACHE_VERSION,
        "repo_id": repo_id,
        "revision": str(revision),
        "source": str(source.source),
        "source_subdir": source.subdir,
        "source_commit": source.commit,
        "source_fingerprint": _source_fingerprint(source),
        "backend": "cpu",
        "torch": str(torch.__version__),
        "python": sys.version,
        "python_abi": sysconfig.get_config_var("SOABI"),
        "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "compiler": {
            "builder": os.environ.get("KERNELS_JIT_BUILDER"),
            "cmake": os.environ.get("KERNELS_JIT_CMAKE"),
            "ninja": os.environ.get("KERNELS_JIT_NINJA"),
            "CC": os.environ.get("CC"),
            "CXX": os.environ.get("CXX"),
            "CFLAGS": os.environ.get("CFLAGS"),
            "CXXFLAGS": os.environ.get("CXXFLAGS"),
            "LDFLAGS": os.environ.get("LDFLAGS"),
            "CMAKE_ARGS": os.environ.get("CMAKE_ARGS"),
            "CMAKE_GENERATOR": os.environ.get("CMAKE_GENERATOR"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:32]


def _source_fingerprint(source: _SourceSpec) -> str | None:
    if not isinstance(source.source, Path):
        return None
    digest = hashlib.sha256()
    for path in sorted(source.source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source.source)
        if any(part in _IGNORE_SOURCE_PARTS for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_cached_variant(final_root: Path) -> Path | None:
    build_root = final_root / "build"
    if not build_root.is_dir():
        return None
    variants = get_variants_local(build_root)
    variant, _ = resolve_variant(variants, "cpu")
    if variant is None:
        return None
    path = build_root / variant.variant_str
    return path if (path / "metadata.json").is_file() else None


def _materialize_source(source: _SourceSpec, destination: Path) -> Path:
    if isinstance(source.source, Path):
        target = destination / "source"
        shutil.copytree(
            source.source,
            target,
            ignore=shutil.ignore_patterns(".git", "build", "_cmake_build", "__pycache__", "*.pyc"),
        )
        return _source_subdir(target, source.subdir)

    if source.commit is None:
        raise JITBuildError(
            "A pinned source commit is required for remote JIT builds. "
            "Pass `jit=JITOptions(revision='<40-character commit>')`."
        )

    parsed = urllib.parse.urlsplit(str(source.source))
    parts = [part for part in parsed.path.split("/") if part]
    owner, repo = parts[0], parts[1].removesuffix(".git")
    tarball = destination / "source.tar.gz"
    url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/{source.commit}"
    request = urllib.request.Request(url, headers={"User-Agent": "kernels-jit"})
    with urllib.request.urlopen(request, timeout=60) as response, tarball.open("wb") as output:
        shutil.copyfileobj(response, output)

    extracted = destination / "extracted"
    extracted.mkdir()
    with tarfile.open(tarball, "r:gz") as archive:
        _safe_extract(archive, extracted)
    entries = [entry for entry in extracted.iterdir() if entry.is_dir()]
    if len(entries) != 1:
        raise JITBuildError(f"Unexpected source archive layout for {url}")
    return _source_subdir(entries[0], source.subdir)


def _source_subdir(root: Path, subdir: str | None) -> Path:
    if subdir is None:
        return root
    path = (root / subdir).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise JITBuildError(f"JIT source subdirectory escapes the source tree: {subdir!r}") from exc
    if not path.is_dir():
        raise JITBuildError(f"JIT source subdirectory does not exist: {path}")
    return path


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        try:
            member_path.relative_to(root)
        except ValueError as exc:
            raise JITBuildError(f"Unsafe path in source archive: {member.name!r}") from exc
        if member.issym() or member.islnk() or member.isdev():
            raise JITBuildError(f"Unsafe entry in source archive: {member.name!r}")
    archive.extractall(destination)


def _run_build(source_root: Path, commit: str | None) -> None:
    builder = _find_tool("KERNELS_JIT_BUILDER", "kernel-builder")
    cmake = _find_tool("KERNELS_JIT_CMAKE", "cmake")
    ninja = _find_tool("KERNELS_JIT_NINJA", "ninja", required=False)
    if builder is None:
        raise JITBuildError(
            "`kernel-builder` is required for JIT builds. Install it with "
            "`pip install hf-kernel-builder` or set KERNELS_JIT_BUILDER."
        )
    if cmake is None:
        raise JITBuildInfoError("`cmake` is required for JIT builds; install cmake >= 3.26.")

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [
            str(Path(sys.executable).parent),
            str(Path(builder).parent),
            env.get("PATH", ""),
        ]
    )
    if ninja is not None:
        env.setdefault("CMAKE_GENERATOR", "Ninja")
        env.setdefault("CMAKE_MAKE_PROGRAM", str(ninja))
    env.setdefault("MAX_JOBS", str(max(1, os.cpu_count() or 1)))

    command = [str(builder), "create-pyproject", "--force"]
    if commit is not None:
        command.extend(["--kernel-sha", commit])
    command.append(str(source_root))
    _run_command(command, cwd=source_root, env=env, description="generate kernel build files")

    _run_command(
        [sys.executable, "setup.py", "build_kernel"],
        cwd=source_root,
        env=env,
        description="compile CPU kernel",
    )


def _find_tool(env_name: str, name: str, *, required: bool = True) -> str | None:
    configured = os.environ.get(env_name)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise JITBuildError(f"{env_name} does not point to a file: {path}")
        return str(path)
    path = shutil.which(name)
    if path is None:
        venv_path = Path(sys.executable).parent / name
        if venv_path.is_file():
            path = str(venv_path)
    if path is None and required:
        return None
    return path


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    description: str,
) -> None:
    logger.info("JIT build: %s (%s)", description, " ".join(command))
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise JITBuildInfoError(f"Could not run {command[0]}: {exc}") from exc
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-40:])
        raise JITBuildError(
            f"JIT build step failed ({description}, exit {result.returncode})." + (f"\n{tail}" if tail else "")
        )


class JITBuildInfoError(JITBuildError):
    """A JIT build failure caused by missing local build prerequisites."""


def _find_built_variant(source_root: Path):
    build_root = source_root / "build"
    variants = get_variants_local(build_root)
    variant, trace = resolve_variant(variants, "cpu")
    if variant is None:
        raise JITBuildInfoError(
            "The JIT build did not produce a CPU variant compatible with this "
            f"environment:\n\n{variants_trace_str(trace)}"
        )
    return build_root / variant.variant_str


def _reject_unsupported_free_threaded_python() -> None:
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        return
    if sys.version_info < (3, 15):
        raise JITBuildInfoError(
            "JIT kernel builds are disabled on free-threaded Python < 3.15. "
            "The current HF kernel C++ ABI is not compatible with this interpreter. "
            "Use a GIL-enabled Python build or a newer free-threaded ABI."
        )


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = [
    "FallbackResolver",
    "JITBuildError",
    "JITBuildInfoError",
    "JITOptions",
    "JITResolver",
]
