from kernels import get_kernel_dep

from . import layers

# Loading must resolve build inputs locally and provide the dependency context.
dependency = get_kernel_dep("kernels-test/symbols-dependency")
assert dependency.VALUE == 42


def relu(x: int) -> int:
    """Apply ReLU."""
    raise AssertionError("Symbol generation must not execute the kernel")


__all__ = ["relu", "layers"]
