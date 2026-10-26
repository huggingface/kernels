import torch

from upstream_relu import _C


def relu(x):
    out = torch.empty_like(x)
    _C.relu(out, x)
    return out
