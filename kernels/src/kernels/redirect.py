import logging
import warnings
from dataclasses import dataclass

from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError

from kernels.compat import tomllib

REDIRECT_FILENAME = "REDIRECT.toml"


@dataclass
class RedirectInfo:
    destination: str
    message: str


def parse_redirect_file(content: str) -> RedirectInfo:
    data = tomllib.loads(content)
    destination = data.get("destination")
    if not destination:
        raise ValueError("REDIRECT.toml must contain a 'destination' field")
    return RedirectInfo(destination=destination, message=data.get("message", ""))


# Check for a redirect file in the repository
def _check_redirect(api: HfApi, repo_id: str, revision: str) -> RedirectInfo | None:
    try:
        path = api.hf_hub_download(
            repo_id=repo_id, filename=REDIRECT_FILENAME, revision=revision
        )
        with open(path, "r") as f:
            return parse_redirect_file(f.read())
    except EntryNotFoundError:
        return None


def resolve_redirect(api: HfApi, repo_id: str, revision: str) -> tuple[str, str]:
    redirect = _check_redirect(api, repo_id, revision)
    if redirect is None:
        return repo_id, revision

    msg = f"WARNING: '{repo_id}' redirected to '{redirect.destination}'"
    if redirect.message:
        msg = f"WARNING: {redirect.message}"

    RED_COLOR = "\033[91m"
    RESET_COLOR = "\033[0m"
    msg = f"\n{RED_COLOR}{msg}{RESET_COLOR}"

    warnings.warn(msg, UserWarning)

    if "@" in redirect.destination:
        return redirect.destination.rsplit("@", 1)
    return redirect.destination, "main"
