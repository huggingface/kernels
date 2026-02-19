import nki
import nki.language as nl
import nki.isa as nisa

from ._ops import ops


@nki.jit(platform_target="trn2")
def relu(x):
    # Check the first dimension's size to ensure it does not exceed on-chip
    # memory tile size, since this simple kernel does not tile inputs.
    assert x.shape[0] <= nl.tile_size.pmax
    x_tile = sbuf.view(dtype=x.dtype, shape=x.shape)
    nisa.dma_copy(dst=x_tile, src=x)
    out_tile = sbuf.view(dtype=x.dtype, shape=x.shape)
    nisa.tensor_scalar(dst=out_tile, data=x_tile, operand0=0, op0=nl.maximum)
    c_output = hbm.view(dtype=x.dtype, shape=x.shape)
    nisa.dma_copy(dst=c_output, src=out_tile)
    return c_output


from . import layers

__all__ = [
    "layers",
    "relu",
]
