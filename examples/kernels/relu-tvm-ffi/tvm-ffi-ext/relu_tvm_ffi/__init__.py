import tvm_ffi

from ._ops import ops


def relu(x: tvm_ffi.Tensor, out: tvm_ffi.Tensor) -> tvm_ffi.Tensor:
    device = x.device
    if device.type == "cpu":
        ops.relu_cpu(out, x)
    elif device.type == "cuda":
        ops.relu_cuda(out, x)
    elif device.type == "xpu":
        ops.relu_xpu(out, x)
    else:
        raise NotImplementedError(f"Unsupported device type: {device.type}")
    return out


from . import layers


__all__ = ["layers", "relu"]
