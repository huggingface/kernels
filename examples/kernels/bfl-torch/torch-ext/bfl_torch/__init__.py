import torch

from ._ops import ops
from ._shader import HEAD_DIM
from .op import _dense_attention  # noqa: F401


def dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Non-causal `softmax(scale * Q K^T) V` per (batch, head).

    Q, K and V are BF16 `[batch, length, heads, 128]` MPS tensors of one
    shape, `length` a multiple of 64. The output has the query's shape.
    `scale` defaults to `128**-0.5`. Each new (length, heads, scale)
    combination compiles the kernel once.

    The kernel requires an M5 or later GPU, macOS 26 or later and torch >=
    2.11; otherwise this falls back to Torch SDPA.
    """
    return ops.dense_attention(
        query, key, value, HEAD_DIM**-0.5 if scale is None else float(scale)
    )


__all__ = ["dense_attention"]
