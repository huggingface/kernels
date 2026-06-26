from .. import relu

try:
    import torch
    from torch import nn

    class ReLU(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            relu(x, out)
            return out

except ImportError:
    pass
