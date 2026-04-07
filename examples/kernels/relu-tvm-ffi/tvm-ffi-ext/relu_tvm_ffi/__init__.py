import tvm_ffi

from ._ops import has_jax, jax_add_ffi_target_name_prefix, ops


def relu(x, out) -> tvm_ffi.Tensor:
    x_t = tvm_ffi.from_dlpack(x)
    out_t = tvm_ffi.from_dlpack(out)

    device = x_t.device
    if device.type == "cpu":
        ops.relu_cpu(out_t, x_t)
    elif device.type == "cuda":
        ops.relu_cuda(out_t, x_t)
    elif device.type == "x_tpu":
        ops.relu_x_tpu(out_t, x)
    else:
        raise NotImplementedError(f"Unsupported device type: {device.type}")

    return out


if has_jax:
    ops_func = (
        getattr(ops, "relu_cuda", None)
        or getattr(ops, "relu_xpu", None)
        or getattr(ops, "relu_cpu", None)
    )

    if ops_func is not None:
        from jax_tvm_ffi import register_ffi_target

        register_ffi_target(
            jax_add_ffi_target_name_prefix("relu"),
            ops_func,
            arg_spec=["rets", "args"],
            platform="cpu" if hasattr(ops, "relu_cpu") else "gpu",
        )


def relu_jax(x):
    if not has_jax:
        raise RuntimeError(
            "JAX is not available. Please install JAX to use this function."
        )

    import jax.ffi

    return jax.ffi.ffi_call(
        jax_add_ffi_target_name_prefix("relu"),
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )(x)


from . import layers


__all__ = ["layers", "relu", "relu_jax"]
