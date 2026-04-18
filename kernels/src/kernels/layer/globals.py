import os
from contextvars import ContextVar
from typing import TYPE_CHECKING, Type

from .repos import DeviceRepos

if TYPE_CHECKING:
    from torch import nn
    from .repos import RepositoryProtocol

_DISABLE_KERNEL_MAPPING: bool = bool(int(os.environ.get("DISABLE_KERNEL_MAPPING", "0")))

_KERNEL_MAPPING: ContextVar[dict[str, dict[str, DeviceRepos]]] = ContextVar("_KERNEL_MAPPING", default={})

_CACHED_LAYER: ContextVar[dict["RepositoryProtocol", Type["nn.Module"]]] = ContextVar("_CACHED_LAYER", default={})
