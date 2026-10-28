import tvm_ffi

from ._ops import ops


def relu(x, out):
    x_t = tvm_ffi.from_dlpack(x)
    out_t = tvm_ffi.from_dlpack(out)

    device = x_t.device
    if device.type == "cuda":
        ops.relu_rust_cuda(x_t, out_t)
    else:
        raise NotImplementedError(f"Unsupported device type: {device.type}")

    return out


__all__ = ["relu"]
