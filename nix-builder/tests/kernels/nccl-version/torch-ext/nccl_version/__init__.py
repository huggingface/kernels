from ._ops import ops


def nccl_header_version() -> int:
    """NCCL version (`NCCL_VERSION_CODE`) that the kernel was compiled against."""
    return ops.nccl_header_version()


def nccl_runtime_version() -> int:
    """Version of the NCCL library that is loaded at runtime."""
    return ops.nccl_runtime_version()


__all__ = ["nccl_header_version", "nccl_runtime_version"]
