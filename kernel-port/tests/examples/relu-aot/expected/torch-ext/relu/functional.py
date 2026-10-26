import torch

from ._ops import ops


def relu(x):
    out = torch.empty_like(x)
    ops.relu(out, x)
    return out
