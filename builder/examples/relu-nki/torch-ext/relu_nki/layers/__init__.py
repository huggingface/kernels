import torch
import torch.nn as nn

from .. import relu


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return relu(x)
