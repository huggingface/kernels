import kernels
import torch


activation = kernels.get_kernel_dep("kernels-community/activation")
einops = kernels.get_kernel_dep("kernels-community/einops")


def silu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    activation.silu(out, x)
    return out


def swap_last_dimensions(x: torch.Tensor) -> torch.Tensor:
    """
    Swap the last two dimensions of a tensor.

    Shapes:
        x: (..., M, N) with at least two dimensions
        return: (..., N, M)

    Raises:
        ValueError: if the tensor has fewer than two dimensions.
    """
    if x.ndim < 2:
        raise ValueError(
            f"Expected a tensor with at least two dimensions, got {x.ndim}"
        )
    return einops.rearrange(x, "... a b -> ... b a")


__all__ = ["swap_last_dimensions"]
