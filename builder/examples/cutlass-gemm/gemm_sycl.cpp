/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief CUTLASS Intel BMG Gemm Example.

    This example constructs and executes a simple CUTLASS GEMM kernel on Intel BMG hardware, and
    verifies its correctness with a reference implementation
    (cutlass::reference::device::GemmComplex). The example also provides a performance measurement
    for the GEMM in TFLOPS.

    This example makes use of BMGs subgroup cooperative 2d-block copy operations and DPAS instructions.

    The shapes of the A and B matrix are defined at runtime by `options.m`, `.n` and `.k`, and the
    batch size is defined by `options.l`. The tile shape, which defines how much work is executed by
    a single work-group, is defined at compile time by:
    ```
      using TileShape = Shape<_256, _256, _32>;
    ```
    That is, each work-group processes a tile of M=256, N=256, and iterates over `options.k` in
    blocks of K=32.

    Performance of GEMM on BMG is heavily dependent on prefetching the A and B matrices. That is,
    executing Intel specific prefetch instructions for future iterations to ensure that the required
    blocks of A and B are resident in cache before they are needed.

    To build & run this example (from your build dir):

      $ ninja 00_bmg_gemm
      $ ./examples/sycl/00_bmg_gemm/00_bmg_gemm

    Call with `--help` for information about available options
*/

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include <torch/all.h>
using namespace cute;

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

void cutlass_gemm(torch::Tensor &out, torch::Tensor const &A, torch::Tensor const &B) {
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = bfloat16_t;
  using ElementInputB = bfloat16_t;
  using ElementOutput = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;
  using TileShape = Shape<_256, _256, _32>;
  using TiledMma = typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;
  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy,
    TileShape,
    ElementAccumulator,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    ElementOutput,
    cutlass::gemm::TagToStrideC_t<LayoutD>,
    FusionCallBacks,
    XE_2D_U32x8x16_LD_N,
    void, void,
    XE_2D_U32x8x16_ST_N,
    void, void>;
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    GEMMDispatchPolicy,
    TileShape,
    ElementInputA,
    cutlass::gemm::TagToStrideA_t<LayoutA>,
    ElementInputB,
    cutlass::gemm::TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, void, void, cute::identity,
    GmemTiledCopyB, void, void, cute::identity>;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  // get shape
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  int L = 1; // batch size

  auto stride_A = cutlass::make_cute_packed_stride(GemmKernel::StrideA{}, cute::make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(GemmKernel::StrideB{}, cute::make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(GemmKernel::StrideC{}, cute::make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(GemmKernel::StrideD{}, cute::make_shape(M, N, L));

  GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    GemmKernel::ProblemShape{M, N, K, L},
    {reinterpret_cast<ElementInputA*>(A.data_ptr()), stride_A, reinterpret_cast<ElementInputB*>(B.data_ptr()), stride_B},
    {{1.0f, 0.0f}, reinterpret_cast<ElementOutput*>(out.data_ptr()), stride_C, reinterpret_cast<ElementOutput*>(out.data_ptr()), stride_D},
    hw_info
  };

  Gemm gemm_op;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  TORCH_CHECK(gemm_op.can_implement(arguments) == cutlass::Status::kSuccess, "Invalid GEMM problem size or configuration");
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
#if defined(OLD_API)
  syclcompat::wait();
#else
  compat::wait();
#endif
}
