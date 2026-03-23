import tvm_ffi

from ._ops import ops


def cutlass_gemm(
    out: tvm_ffi.Tensor, A: tvm_ffi.Tensor, B: tvm_ffi.Tensor
) -> tvm_ffi.Tensor:
    device = A.device
    if device.type == "cuda":
        ops.cutlass_gemm(out, A, B)
    elif device.type == "xpu":
        ops.cutlass_gemm(out, A, B)
    else:
        raise NotImplementedError(f"Unsupported device type: {device.type}")
    return out


__all__ = ["cutlass_gemm"]
