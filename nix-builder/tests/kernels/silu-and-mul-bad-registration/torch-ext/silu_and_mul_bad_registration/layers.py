import torch
import torch.nn as nn

from ._ops import ops


class SiluAndMul(nn.Module):
    """
    Apply SiLU to one half of the array and use it as a multiplicative
    gate for the other half.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor):
        return ops.silu_and_mul(x)
