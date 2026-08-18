import re

from kernels_data import Backend, Metadata

from kernels.compat import has_torch


class UnsupportedArchError(RuntimeError):
    """The kernel build does not support the architecture of the current device."""


# CUDA archs are compute capabilities like `9.0`, optionally with an
# architecture-specific (`9.0a`) or family-specific (`10.0f`) suffix.
_CUDA_ARCH_REGEX = re.compile(r"(\d+)\.(\d+)([af]?)")


def _cuda_arch_supports(arch: str, capability: tuple[int, int]) -> bool | None:
    """Check whether a single declared CUDA arch supports a compute capability.

    Returns `None` when the arch string is not in a known format.
    """
    m = _CUDA_ARCH_REGEX.fullmatch(arch)
    if m is None:
        return None

    arch_major, arch_minor, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
    major, minor = capability

    if suffix == "a":
        # Architecture-specific builds only run on that exact capability.
        return (major, minor) == (arch_major, arch_minor)

    # Base and family-specific builds run on capabilities of the same
    # generation with the same or a newer minor version.
    return major == arch_major and minor >= arch_minor


def _cuda_archs_support_capability(archs: list[str], capability: tuple[int, int]) -> bool:
    supports = [_cuda_arch_supports(arch, capability) for arch in archs]
    if all(support is None for support in supports):
        # None of the arch strings are in a known format (e.g. produced by a
        # newer kernel-builder), so compatibility cannot be determined.
        return True
    return any(supports)


def _arch_incompatibility(metadata: Metadata) -> str | None:
    """Check a kernel build against the architecture of the current device."""
    archs = metadata.backend.archs
    if not archs or not has_torch:
        return None

    import torch

    backend_type = metadata.backend.backend_type
    if backend_type == Backend.CUDA:
        if torch.version.cuda is None or not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
        if _cuda_archs_support_capability(archs, (major, minor)):
            return None
        return (
            f"CUDA capability {major}.{minor} of the current device is not "
            f"supported by the architectures of the build: {', '.join(archs)}"
        )
    elif backend_type == Backend.ROCm:
        if torch.version.hip is None or not torch.cuda.is_available():
            return None
        gcn_arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
        # Strip feature flags, e.g. `gfx90a:sramecc+:xnack-` -> `gfx90a`.
        gcn_arch = gcn_arch.split(":")[0]
        if gcn_arch in archs:
            return None
        return (
            f"ROCm arch {gcn_arch} of the current device is not supported "
            f"by the architectures of the build: {', '.join(archs)}"
        )

    return None


def check_arch_compatibility(metadata: Metadata, variant_str: str) -> None:
    """Raise `UnsupportedArchError` when the build does not support the current device."""
    reason = _arch_incompatibility(metadata)
    if reason is not None:
        raise UnsupportedArchError(
            f"Kernel '{metadata.name}' variant '{variant_str}' does not support "
            f"the current device: {reason}. The declared architectures can be "
            "incomplete (e.g. when a kernel has a Triton fallback), pass "
            "`check_arch=False` to skip this check. Warning: the kernel may "
            "fail or crash the process at run time when it does not support "
            "the device."
        )


def is_arch_compatible(metadata: Metadata) -> bool:
    """Check whether the build supports the architecture of the current device."""
    return _arch_incompatibility(metadata) is None
