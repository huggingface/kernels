import platform
from functools import lru_cache

import mlx.core as mx

from ._shader import HEAD_DIM, HEADER, SOURCE, THREADS_PER_THREADGROUP, launch_shape

# The shader assumes the cooperative tensor layout of GPU generation 17 (M5) and later.
_MIN_GPU_GENERATION = 17

# The shader uses Metal 4 (MetalPerformancePrimitives), which requires macOS 26.
_MIN_MACOS_MAJOR = 26


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="dense_attention_bf16_d128",
        input_names=["q", "k", "v"],
        output_names=["o"],
        header=HEADER,
        source=SOURCE,
    )


@lru_cache(maxsize=1)
def _shader_supported() -> bool:
    # The architecture is e.g. "applegpu_g17s".
    if int(mx.device_info()["architecture"][-3:-1]) < _MIN_GPU_GENERATION:
        return False
    # `mac_ver()` is e.g. "26.0.1", or empty if the version is unknown.
    macos_major = platform.mac_ver()[0].split(".")[0]
    return macos_major.isdigit() and int(macos_major) >= _MIN_MACOS_MAJOR


def dense_attention(
    query: mx.array, key: mx.array, value: mx.array, scale: float | None = None
) -> mx.array:
    """Non-causal `softmax(scale * Q K^T) V` per (batch, head).

    Q, K and V are BF16 `[batch, length, heads, 128]` arrays of one shape,
    `length` a multiple of 64. The output has the query's shape. `scale`
    defaults to `128**-0.5`. Each new (length, heads, scale) combination
    compiles the kernel once.

    The kernel requires an M5 or later GPU and macOS 26 or later; otherwise
    this falls back to MLX SDPA.
    """
    scale = HEAD_DIM**-0.5 if scale is None else float(scale)
    for name, a in (("query", query), ("key", key), ("value", value)):
        if a.dtype != mx.bfloat16:
            raise TypeError(f"{name} must be bfloat16, got {a.dtype}")
        if a.shape != query.shape:
            raise ValueError(f"{name} shape {a.shape} != query {query.shape}")
    threadgroups, scale2_bits = launch_shape(tuple(query.shape), scale)
    if threadgroups == 0:
        return mx.zeros(query.shape, dtype=mx.bfloat16)
    if not _shader_supported():
        # MLX SDPA wants [B, H, L, D].
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (query, key, value))
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale).transpose(
            0, 2, 1, 3
        )
    _, seq, heads, _ = query.shape
    return _kernel()(
        inputs=[query, key, value],
        template=[("SEQ", seq), ("HEADS", heads), ("SCALE2_BITS", scale2_bits)],
        grid=(threadgroups * THREADS_PER_THREADGROUP, 1, 1),
        threadgroup=(THREADS_PER_THREADGROUP, 1, 1),
        output_shapes=[query.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


__all__ = ["dense_attention"]
