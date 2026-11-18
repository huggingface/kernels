from functools import lru_cache

import mlx.core as mx

from ._shader import HEAD_DIM, HEADER, SOURCE, THREADS_PER_THREADGROUP, launch_shape


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="dense_attention_bf16_d128",
        input_names=["q", "k", "v"],
        output_names=["o"],
        header=HEADER,
        source=SOURCE,
    )


def dense_attention(query: mx.array, key: mx.array, value: mx.array, scale: float | None = None) -> mx.array:
    """Non-causal `softmax(scale * Q K^T) V` per (batch, head).

    Q, K and V are BF16 `[batch, length, heads, 128]` arrays of one shape,
    `length` a multiple of 64. The output has the query's shape. `scale`
    defaults to `128**-0.5`. Each new (length, heads, scale) combination
    compiles the kernel once.
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
