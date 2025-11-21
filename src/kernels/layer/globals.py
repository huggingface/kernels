import os
from typing import (
    Dict,
)
from contextvars import ContextVar
from .repos import DeviceRepos


_DISABLE_KERNEL_MAPPING: bool = bool(int(os.environ.get("DISABLE_KERNEL_MAPPING", "0")))

_KERNEL_MAPPING: ContextVar[Dict[str, Dict[str, DeviceRepos]]] = ContextVar(
    "_KERNEL_MAPPING", default={}
)
