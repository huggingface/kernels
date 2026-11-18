"""Dense BF16 attention on Apple GPUs: one Metal kernel, served to torch and MLX.

The same Metal Shading Language source (``HEADER`` + ``SOURCE`` below) runs as

* a torch MPS op: ``torch.mps.compile_shader`` plus ``torch.library.custom_op``,
  so it takes torch MPS tensors, runs on torch's own queue and stays a single
  node under ``torch.compile(fullgraph=True)``;
* an MLX op: ``mx.fast.metal_kernel`` on ``mx.array`` inputs.

Contract: non-causal ``softmax(scale * Q K^T) V`` per (batch, head). Q, K and V
are BF16 ``[batch, length, heads, 128]`` of one shape, ``length`` a multiple of
64, no mask. The output has the query's shape. ``SEQ``, ``HEADS`` and the
scale are compile-time constants, so each new combination compiles once
(about 0.2 s) and is cached.

Kernel structure: a threadgroup is 4 simdgroups x 16 query rows of one
(batch, head); 32 keys per step; every MMA is a 16x32x16 MetalPerformancePrimitives
``matmul2d`` (Metal 4); online softmax in base 2 with ``scale * log2(e)``
folded into one constant. Written by Black Forest Labs (kirschkern), following
the structure of MLX's NAX SDPA forward; no MLX source is copied.

Requirements: Apple silicon with Metal 4, torch >= 2.11 (its ``compile_shader``
targets Metal 4.0) and/or mlx (tested 0.32.4). Tested only on an M5 Ultra,
macOS 27, torch 2.15.0.

Usage: python dense_attention_demo.py [--batch 1 --seq 4608 --heads 24]
"""

from __future__ import annotations

import argparse
import math
import statistics
import struct
import time
from functools import lru_cache

import torch

HEAD_DIM = 128
ROWS_PER_THREADGROUP = 64
THREADS_PER_THREADGROUP = 128
LOG2E = 1.4426950408889634

HEADER = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

typedef vec<float, 8> ffrag;
typedef vec<bfloat, 8> bfrag;

// Lane coordinates in a 16x16 fragment: x = column, y = row. Lane l holds rows
// fm + {0, 8} and columns fn + {0..3}.
inline short2 frag_coord(ushort lane) {
  const short fm = (((lane >> 4) & 1) << 2) | ((lane >> 1) & 3);
  const short fn = ((((lane >> 3) & 1) << 1) | (lane & 1)) * 4;
  return short2{fn, fm};
}

// Load a 16x16 fragment whose rows are `stride` elements apart.
template <typename P>
inline void frag_load(thread bfrag& dst, P src, int stride, short2 sc) {
  src += sc.y * stride + sc.x;
  #pragma clang loop unroll(full)
  for (short i = 0; i < 2; i++) {
    #pragma clang loop unroll(full)
    for (short j = 0; j < 4; j++) dst[i * 4 + j] = src[(i * 8) * stride + j];
  }
}

// C[16x32] += A[16x16] * B[16x32], C and B split into two 16x16 fragments
// along N. TB: the B fragments hold B^T (rows = N). Relaxed precision.
template <bool TB, typename AT>
inline void mma_n2(thread ffrag& C0, thread ffrag& C1, const thread vec<AT, 8>& A,
                   const thread bfrag& B0, const thread bfrag& B1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, TB, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<AT, bfloat, float>();
  auto cb = op.template get_right_input_cooperative_tensor<AT, bfloat, float>();
  auto cc = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) ca[i] = A[i];
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) { cb[i] = B0[i]; cb[8 + i] = B1[i]; }
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) { cc[i] = C0[i]; cc[8 + i] = C1[i]; }
  op.run(ca, cb, cc);
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) { C0[i] = cc[i]; C1[i] = cc[8 + i]; }
}

// Per-lane partial over row slot r (elements 4r..4r+3), reduced over the row's 4 lanes.
template <bool MAX>
inline float frag_row_reduce(const thread ffrag& f, short r) {
  float t = MAX ? fmax(fmax(f[4 * r], f[4 * r + 1]), fmax(f[4 * r + 2], f[4 * r + 3]))
                : (f[4 * r] + f[4 * r + 1]) + (f[4 * r + 2] + f[4 * r + 3]);
  float u = simd_shuffle_xor(t, ushort(1));
  t = MAX ? fmax(t, u) : t + u;
  u = simd_shuffle_xor(t, ushort(8));
  return MAX ? fmax(t, u) : t + u;
}
"""

SOURCE = r"""
  constexpr int D = 128;
  constexpr int HD = HEADS * D;  // token stride of [B, L, H, D]
  constexpr int WM = 4;          // simdgroups per TG, 16 query rows each
  constexpr int BQ = 16 * WM;
  constexpr int BK = 32;         // keys per step
  constexpr int NQB = SEQ / BQ;
  constexpr int TV = 8;          // D fragments of O
  constexpr int QS = D + 8;      // padded tgmem row stride of Q: no bank conflicts
  const float scale2 = as_type<float>(SCALE2_BITS);
  const ushort lane = thread_index_in_simdgroup;
  const int sg = simdgroup_index_in_threadgroup;
  const int lin = threadgroup_position_in_grid.x;
  const int qb = lin % NQB;      // query blocks of one (batch, head) fastest
  const int bh = lin / NQB;
  const long base = (long)(bh / HEADS) * SEQ * HD + (bh % HEADS) * D;
  const short2 sc = frag_coord(lane);

  // Stage the TG's BQ x 128 Q tile once (uint4 loads); Q device reads would
  // compete with the K stream.
  threadgroup bfloat qs[BQ * QS];
  {
    const uint tid = thread_index_in_threadgroup;
    const device bfloat* qtile = (const device bfloat*)q + base + (long)(qb * BQ) * HD;
    for (uint e = tid * 8; e < BQ * D; e += WM * 32 * 8)
      *(threadgroup uint4*)(qs + (e / D) * QS + (e % D)) =
          *(const device uint4*)(qtile + (long)(e / D) * HD + (e % D));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const threadgroup bfloat* Qs = qs + sg * 16 * QS;
  const device bfloat* Kp = (const device bfloat*)k + base;
  const device bfloat* Vp = (const device bfloat*)v + base;

  ffrag Ot[TV];
  #pragma clang loop unroll(full)
  for (short i = 0; i < TV; i++) Ot[i] = ffrag(0.0f);
  float max_score[2] = {-FLT_MAX, -FLT_MAX};
  float sum_score[2] = {0.0f, 0.0f};

  for (int kb = 0; kb < SEQ / BK; kb++) {
    // S[16 x 32] = Q K^T over 8 D fragments; unroll 4 so K loads interleave with the MMAs.
    ffrag St[2] = {ffrag(0.0f), ffrag(0.0f)};
    #pragma clang loop unroll_count(4)
    for (short id = 0; id < 8; id++) {
      bfrag Qf, K0, K1;
      frag_load(Qf, Qs + id * 16, QS, sc);
      frag_load(K0, Kp + id * 16, HD, sc);
      frag_load(K1, Kp + 16 * HD + id * 16, HD, sc);
      mma_n2<true>(St[0], St[1], Qf, K0, K1);
    }

    // Online softmax, base 2; O and l rescaled every step.
    #pragma clang loop unroll(full)
    for (short f = 0; f < 2; f++) St[f] *= scale2;
    float new_max[2] = {max_score[0], max_score[1]};
    #pragma clang loop unroll(full)
    for (short f = 0; f < 2; f++) {
      new_max[0] = fmax(new_max[0], frag_row_reduce<true>(St[f], 0));
      new_max[1] = fmax(new_max[1], frag_row_reduce<true>(St[f], 1));
    }
    float factor[2];
    #pragma clang loop unroll(full)
    for (short r = 0; r < 2; r++) {
      factor[r] = fast::exp2(max_score[r] - new_max[r]);
      max_score[r] = new_max[r];
      sum_score[r] *= factor[r];
    }
    #pragma clang loop unroll(full)
    for (short i = 0; i < TV; i++) {
      #pragma clang loop unroll(full)
      for (short e = 0; e < 8; e++) Ot[i][e] *= factor[e >> 2];
    }
    #pragma clang loop unroll(full)
    for (short f = 0; f < 2; f++) {
      #pragma clang loop unroll(full)
      for (short e = 0; e < 8; e++) St[f][e] = fast::exp2(St[f][e] - max_score[e >> 2]);
    }
    #pragma clang loop unroll(full)
    for (short f = 0; f < 2; f++) {
      sum_score[0] += frag_row_reduce<false>(St[f], 0);
      sum_score[1] += frag_row_reduce<false>(St[f], 1);
    }
    simdgroup_barrier(mem_flags::mem_none);

    // O += P V: P is the FP32 S fragment (float x bfloat), V tiles of 16 keys x 32 columns.
    #pragma clang loop unroll(full)
    for (short id = 0; id < TV; id += 2) {
      if (id == 4) threadgroup_barrier(mem_flags::mem_none);  // removing it costs 7%
      #pragma clang loop unroll(full)
      for (short ik = 0; ik < 2; ik++) {
        bfrag V0, V1;
        frag_load(V0, Vp + ik * 16 * HD + id * 16, HD, sc);
        frag_load(V1, Vp + ik * 16 * HD + id * 16 + 16, HD, sc);
        mma_n2<false>(Ot[id], Ot[id + 1], St[ik], V0, V1);
      }
    }
    Kp += BK * HD;
    Vp += BK * HD;
  }

  threadgroup_barrier(mem_flags::mem_none);
  const float rcp[2] = {1.0f / sum_score[0], 1.0f / sum_score[1]};
  device bfloat* Op = (device bfloat*)o + base + (long)(qb * BQ + sg * 16 + sc.y) * HD + sc.x;
  #pragma clang loop unroll(full)
  for (short i = 0; i < TV; i++) {
    #pragma clang loop unroll(full)
    for (short r = 0; r < 2; r++) {
      #pragma clang loop unroll(full)
      for (short j = 0; j < 4; j++)
        Op[(r * 8) * HD + i * 16 + j] = static_cast<bfloat>(Ot[i][r * 4 + j] * rcp[r]);
    }
  }
"""


def _launch_shape(shape: tuple[int, ...], scale: float) -> tuple[int, int]:
    """Validate a [B, L, H, 128] shape; return (threadgroups, FP32 bits of scale*log2(e))."""
    if len(shape) != 4 or shape[3] != HEAD_DIM:
        raise ValueError(f"expected [batch, length, heads, {HEAD_DIM}], got {tuple(shape)}")
    b, seq, heads, _ = shape
    if seq % ROWS_PER_THREADGROUP:
        raise ValueError(f"length={seq} must be a multiple of {ROWS_PER_THREADGROUP}")
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be finite and positive, got {scale}")
    (scale2_bits,) = struct.unpack("<i", struct.pack("<f", scale * LOG2E))
    return b * heads * (seq // ROWS_PER_THREADGROUP), scale2_bits


# ---------------------------------------------------------------- torch (MPS)

# compile_shader takes a complete kernel, so the body gets a signature here;
# MLX's metal_kernel generates the equivalent one from input/output names.
TORCH_SIGNATURE = """
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
def _torch_library(seq: int, heads: int, scale2_bits: int):
    defines = f"#define SEQ {seq}\n#define HEADS {heads}\n#define SCALE2_BITS {scale2_bits}\n"
    return torch.mps.compile_shader(HEADER + defines + TORCH_SIGNATURE % SOURCE)


@torch.library.custom_op("bfl::dense_attention", mutates_args=(), device_types="mps")
def _dense_attention_op(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float
) -> torch.Tensor:
    for name, t in (("query", query), ("key", key), ("value", value)):
        if t.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16, got {t.dtype}")
        if t.shape != query.shape:
            raise ValueError(f"{name} shape {tuple(t.shape)} != query {tuple(query.shape)}")
    threadgroups, scale2_bits = _launch_shape(tuple(query.shape), scale)
    q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
    out = torch.empty_like(q)
    if threadgroups:
        lib = _torch_library(q.shape[1], q.shape[2], scale2_bits)
        lib.dense_attention_bf16_d128(
            q, k, v, out,
            threads=[threadgroups * THREADS_PER_THREADGROUP],
            group_size=[THREADS_PER_THREADGROUP],
        )
    return out


@_dense_attention_op.register_fake
def _(query, key, value, scale):
    return torch.empty_like(query)


def dense_attention_torch(query, key, value, scale: float | None = None):
    """torch MPS entry point: BF16 [B, L, H, 128] tensors in, a new tensor out.

    Also callable as ``torch.ops.bfl.dense_attention(q, k, v, scale)``.
    """
    return _dense_attention_op(query, key, value, HEAD_DIM**-0.5 if scale is None else float(scale))


# ------------------------------------------------------------------------ MLX


@lru_cache(maxsize=1)
def _mlx_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="dense_attention_bf16_d128",
        input_names=["q", "k", "v"],
        output_names=["o"],
        header=HEADER,
        source=SOURCE,
    )


def dense_attention_mlx(query, key, value, scale: float | None = None):
    """MLX entry point: BF16 [B, L, H, 128] mx.arrays in, a new mx.array out."""
    import mlx.core as mx

    scale = HEAD_DIM**-0.5 if scale is None else float(scale)
    for name, a in (("query", query), ("key", key), ("value", value)):
        if a.dtype != mx.bfloat16:
            raise TypeError(f"{name} must be bfloat16, got {a.dtype}")
        if a.shape != query.shape:
            raise ValueError(f"{name} shape {a.shape} != query {query.shape}")
    threadgroups, scale2_bits = _launch_shape(tuple(query.shape), scale)
    if threadgroups == 0:
        return mx.zeros(query.shape, dtype=mx.bfloat16)
    _, seq, heads, _ = query.shape
    return _mlx_kernel()(
        inputs=[query, key, value],
        template=[("SEQ", seq), ("HEADS", heads), ("SCALE2_BITS", scale2_bits)],
        grid=(threadgroups * THREADS_PER_THREADGROUP, 1, 1),
        threadgroup=(THREADS_PER_THREADGROUP, 1, 1),
        output_shapes=[query.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


# ----------------------------------------------------------------------- demo


def _rel_rms(x, ref) -> float:
    return ((x.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def _ms_per_call(step, x0, sync, n: int = 20, repeats: int = 7) -> float:
    """Median ms per call of a dependent chain x = step(x), so calls cannot overlap."""

    def run(count):
        x = x0
        for _ in range(count):
            x = step(x)
        sync(x)

    run(3)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run(n)
        times.append((time.perf_counter() - t0) / n * 1e3)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=4608)
    parser.add_argument("--heads", type=int, default=24)
    args = parser.parse_args()

    import torch.nn.functional as F

    torch.manual_seed(0)
    shape = (args.batch, args.seq, args.heads, HEAD_DIM)
    q, k, v = (torch.randn(shape, device="mps", dtype=torch.bfloat16) for _ in range(3))
    print(f"shape [B, L, H, D] = {list(shape)}, torch {torch.__version__}")

    # torch SDPA wants [B, H, L, D]; the FP32 run is the accuracy reference.
    def sdpa(q, k, v):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    ref = sdpa(q.float(), k.float(), v.float())

    t0 = time.perf_counter()
    out = dense_attention_torch(q, k, v)
    torch.mps.synchronize()
    print(f"torch: first call (compile + run) {time.perf_counter() - t0:.3f} s")
    print(f"torch: rel_rms vs FP32 SDPA: ours {_rel_rms(out, ref):.3e}, "
          f"torch BF16 SDPA {_rel_rms(sdpa(q, k, v), ref):.3e}")

    compiled = torch.compile(lambda q, k, v: dense_attention_torch(q, k, v) * 2.0, fullgraph=True)
    same = torch.equal(compiled(q, k, v), out * 2.0)
    print(f"torch: torch.compile(fullgraph=True) runs, matches eager bitwise: {same}")

    # Each call's output is the next call's query (same shape, values stay in range).
    sync = lambda x: torch.mps.synchronize()
    ours_ms = _ms_per_call(lambda x: dense_attention_torch(x, k, v), q, sync)
    sdpa_ms = _ms_per_call(lambda x: sdpa(x, k, v), q, sync)
    print(f"torch: ms/call: ours {ours_ms:.3f}, F.scaled_dot_product_attention {sdpa_ms:.3f}")

    try:
        import mlx.core as mx
    except ImportError:
        print("mlx: not installed, skipped")
        return

    qm, km, vm = (mx.array(t) for t in (q, k, v))  # DLPack: copies into MLX
    out_mlx = dense_attention_mlx(qm, km, vm)
    mx.eval(out_mlx)
    same = torch.equal(torch.from_dlpack(out_mlx), out)  # zero-copy view of the MLX buffer
    print(f"mlx {mx.__version__}: output bitwise equal to the torch op: {same}")

    def mlx_sdpa(q, k, v):
        t = lambda a: a.transpose(0, 2, 1, 3)
        return t(mx.fast.scaled_dot_product_attention(t(q), t(k), t(v), scale=HEAD_DIM**-0.5))

    print(f"mlx: rel_rms of mx.fast SDPA vs FP32 SDPA: "
          f"{_rel_rms(torch.from_dlpack(mx.contiguous(mlx_sdpa(qm, km, vm))), ref):.3e}")
    mlx_ms = _ms_per_call(lambda x: dense_attention_mlx(x, km, vm), qm, mx.eval)
    mlx_sdpa_ms = _ms_per_call(lambda x: mlx_sdpa(x, km, vm), qm, mx.eval)
    print(f"mlx: ms/call: ours {mlx_ms:.3f}, mx.fast.scaled_dot_product_attention {mlx_sdpa_ms:.3f}")


if __name__ == "__main__":
    main()