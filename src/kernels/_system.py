import platform
from typing import Optional


def glibc_version() -> Optional[str]:
    libc_version = platform.libc_ver()

    if len(libc_version) == 2 and libc_version[0] == "glibc":
        return libc_version[1]

    return None
