from functools import lru_cache

import torch

from ._ops import add_op_namespace_prefix
from ._shader import HEADER, SOURCE, THREADS_PER_THREADGROUP, launch_shape

# `torch.mps.compile_shader` targets Metal 4.0 since this Torch version.
_MIN_TORCH_VERSION = (2, 11)

# `compile_shader` takes a complete kernel, so the body gets a signature here.
_SIGNATURE = """
kernel void dense_attention_bf16_d128(
    device bfloat* q [[buffer(0)]], device bfloat* k [[buffer(1)]],
    device bfloat* v [[buffer(2)]], device bfloat* o [[buffer(3)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]) {
%s
}
"""


@lru_cache(maxsize=None)
def _library(seq: int, heads: int, scale2_bits: int):
    torch_version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if torch_version < _MIN_TORCH_VERSION:
        raise RuntimeError(
            f"dense_attention requires torch >= {'.'.join(map(str, _MIN_TORCH_VERSION))}, got {torch.__version__}"
        )
    defines = f"#define SEQ {seq}\n#define HEADS {heads}\n#define SCALE2_BITS {scale2_bits}\n"
    return torch.mps.compile_shader(HEADER + defines + _SIGNATURE % SOURCE)


@torch.library.custom_op(add_op_namespace_prefix("dense_attention"), mutates_args=(), device_types="mps")
def _dense_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
    for name, t in (("query", query), ("key", key), ("value", value)):
        if t.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16, got {t.dtype}")
        if t.shape != query.shape:
            raise ValueError(f"{name} shape {tuple(t.shape)} != query {tuple(query.shape)}")
    threadgroups, scale2_bits = launch_shape(tuple(query.shape), scale)
    q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
    out = torch.empty_like(q)
    if threadgroups:
        lib = _library(q.shape[1], q.shape[2], scale2_bits)
        lib.dense_attention_bf16_d128(
            q,
            k,
            v,
            out,
            threads=[threadgroups * THREADS_PER_THREADGROUP],
            group_size=[THREADS_PER_THREADGROUP],
        )
    return out


@_dense_attention.register_fake
def _(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.empty_like(query)
