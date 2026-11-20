"""Metal Shading Language source of the dense BF16 attention kernel.

`HEADER` and `SOURCE` are the kernel's helpers and body. The same source is
used by the MLX and the Torch variant of this kernel; `SEQ`, `HEADS` and
`SCALE2_BITS` are supplied by the framework wrapper as compile-time constants.

Kernel structure: a threadgroup is 4 simdgroups x 16 query rows of one
(batch, head); 32 keys per step; every MMA is a 16x32x16 MetalPerformancePrimitives
``matmul2d`` (Metal 4); online softmax in base 2 with ``scale * log2(e)``
folded into one constant. Written by Black Forest Labs (kirschkern), following
the structure of MLX's NAX SDPA forward; no MLX source is copied.
"""

import math
import struct

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


def launch_shape(shape: tuple[int, ...], scale: float) -> tuple[int, int]:
    """Validate a [B, L, H, 128] shape; return (threadgroups, FP32 bits of scale*log2(e))."""
    if len(shape) != 4 or shape[3] != HEAD_DIM:
        raise ValueError(
            f"expected [batch, length, heads, {HEAD_DIM}], got {tuple(shape)}"
        )
    b, seq, heads, _ = shape
    if seq % ROWS_PER_THREADGROUP:
        raise ValueError(f"length={seq} must be a multiple of {ROWS_PER_THREADGROUP}")
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be finite and positive, got {scale}")
    (scale2_bits,) = struct.unpack("<i", struct.pack("<f", scale * LOG2E))
    return b * heads * (seq // ROWS_PER_THREADGROUP), scale2_bits
