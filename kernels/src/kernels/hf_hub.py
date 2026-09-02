import os
import platform
import warnings
from dataclasses import dataclass

from huggingface_hub import HfApi, constants

from kernels._system import glibc_version
from kernels.backends import _select_backend
from kernels.compat import has_torch, has_tvm_ffi


def _platform() -> str:
    cpu = platform.machine()
    os = platform.system().lower()

    if os == "darwin":
        cpu = "aarch64" if cpu == "arm64" else cpu
    elif os == "windows":
        cpu = "x86_64" if cpu == "AMD64" else cpu

    return f"{cpu}-{os}"


def _get_hf_api(user_agent: str | dict | None = None) -> HfApi:
    """Returns an instance of HfApi with proper settings."""

    from . import __version__

    user_agent_str = ""
    if not constants.HF_HUB_DISABLE_TELEMETRY:
        parts: list[str] = []

        # User-defined info
        if isinstance(user_agent, dict):
            parts.extend(f"{k}/{v}" for k, v in user_agent.items())
        elif isinstance(user_agent, str) and user_agent:
            parts.append(user_agent)

        # System info
        python = ".".join(platform.python_version_tuple()[:2])
        backend = _select_backend(None).variant_str
        parts.extend(
            [
                f"kernels/{__version__}",
                f"python/{python}",
                f"backend/{backend}",
                f"platform/{_platform()}",
                "file_type/kernel",
            ]
        )

        if has_torch:
            import torch

            parts.append(f"torch/{torch.__version__}")
        if has_tvm_ffi:
            import tvm_ffi

            parts.append(f"tvm-ffi/{tvm_ffi.__version__}")

        # Add glibc version if available
        glibc = glibc_version()
        if glibc is not None:
            parts.append(f"glibc/{glibc}")

        user_agent_str = "; ".join(parts)

    return HfApi(library_name="kernels", library_version=__version__, user_agent=user_agent_str)


def _get_cache_dir() -> str | None:
    """Returns the kernels cache directory."""
    return os.environ.get("KERNELS_CACHE", None)


CACHE_DIR: str | None = _get_cache_dir()


@dataclass(frozen=True)
class RepoInfo:
    """
    This dataclass stores the origin of the kernel.

    The following fields are available:

    - `repo_id` (`str`): the Hub repository containing the kernel.
    - `revision` (`str`): the specific revision of the kernel.
    """

    repo_id: str
    revision: str


def _check_trust_remote_code(repo_id: str, local_files_only: bool, trust_remote_code: bool | list[str]) -> None:
    """Check whether a kernel repository is trusted.

    When ``trust_remote_code`` is ``False`` (the default), only repositories
    whose publisher organization has ``trustedKernelPublisher`` enabled on the
    Hub are allowed. Repositories from untrusted publishers will raise a
    ``ValueError``.

    When ``trust_remote_code`` is ``True``, all repositories are allowed.

    When ``trust_remote_code`` is a list of strings, it is treated as an
    allowlist of repository IDs. Only repositories in the list and repositories
    from trusted publishers are allowed.
    """
    if trust_remote_code is True:
        return

    if isinstance(trust_remote_code, list) and repo_id in trust_remote_code:
        return

    if local_files_only:
        # Publisher trust cannot be verified offline. The user opted into
        # offline mode and the kernel must already be in the local cache,
        # so trust was established when it was originally downloaded.
        warnings.warn(
            f"Skipping publisher trust check for '{repo_id}' because Hugging Face Hub is in offline mode.",
            stacklevel=3,
        )
        return

    publisher = repo_id.split("/", 1)[0]

    try:
        info = _get_hf_api().get_organization_overview(publisher)
    except Exception:
        raise ValueError(
            f"Kernel repository '{repo_id}' could not verify publisher trust status. "
            "Set trust_remote_code=True or add the repository ID to the trust_remote_code allowlist "
            "to allow loading kernels from untrusted sources."
        )

    if getattr(info, "trustedKernelPublisher", False):
        return

    raise ValueError(
        f"Kernel repository '{repo_id}' is not from a trusted publisher. "
        "Set trust_remote_code=True or add the repository ID to the trust_remote_code allowlist "
        "to allow loading kernels from untrusted sources."
    )
