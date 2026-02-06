import torch
import torch.nn as nn

from .._ops import ops


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        ops.relu(out, x)
        return out
