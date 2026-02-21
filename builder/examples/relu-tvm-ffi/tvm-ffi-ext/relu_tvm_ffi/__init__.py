import tvm_ffi

from ._ops import ops


def relu(x: tvm_ffi.Tensor, out: tvm_ffi.Tensor) -> tvm_ffi.Tensor:
    device = x.device
    if device.type == "cuda":
        ops.relu_cuda(out, x)
    else:
        raise NotImplementedError(f"Unsupported device type: {device.type}")
    return out


__all__ = ["relu"]
