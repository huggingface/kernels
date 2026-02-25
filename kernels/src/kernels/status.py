import warnings
from dataclasses import dataclass
from typing import Union

from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError

from kernels.compat import tomllib


@dataclass
class Redirect:
    kind: str  # must be "redirect"
    destination: str
    revision: str

    @staticmethod
    def from_dict(data: dict) -> "Redirect":
        if data.get("kind") != "redirect":
            raise ValueError("kernel-status.toml kind must be 'redirect' for Redirect")
        destination = data.get("destination")
        if not destination:
            raise ValueError("kernel-status.toml must contain a 'destination' field")
        return Redirect(
            kind="redirect",
            destination=destination,
            revision=data.get("revision", "main"),
        )


KernelStatusKind = Union[Redirect]


class KernelStatus:
    @staticmethod
    def from_toml(content: str) -> KernelStatusKind:
        data = tomllib.loads(content)

        kind = data.get("kind")
        if not kind:
            raise ValueError("kernel-status.toml must contain a 'kind' field")

        if kind == "redirect":
            return Redirect.from_dict(data)

        raise ValueError(f"Unknown kernel status kind: {kind!r}")

    # Fetch the kernel status from the repository, if it exists
    @staticmethod
    def check_status(
        api: HfApi, repo_id: str, revision: str
    ) -> KernelStatusKind | None:
        try:
            path = api.hf_hub_download(
                repo_id=repo_id, filename="kernel-status.toml", revision=revision
            )
            with open(path, "r") as f:
                return KernelStatus.from_toml(f.read())
        except EntryNotFoundError:
            return None


def resolve_status(api: HfApi, repo_id: str, revision: str) -> tuple[str, str]:
    status = KernelStatus.check_status(api, repo_id, revision)
    if status is None:
        return repo_id, revision

    # In the case of a redirect, return the destination repo and revision
    if isinstance(status, Redirect):
        warnings.warn(
            f"'{repo_id}' redirected to '{status.destination}'",
            UserWarning,
            stacklevel=2,
        )
        return status.destination, status.revision

    return repo_id, revision
