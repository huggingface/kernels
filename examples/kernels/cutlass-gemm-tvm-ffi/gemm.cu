#include <cutlass/gemm/device/gemm.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/extra/c_env_api.h>

#include "util.hh"

using namespace tvm;

void cutlass_gemm(ffi::TensorView out, ffi::TensorView const A, ffi::TensorView const B) {
    CHECK_INPUT_CUDA(A);
    CHECK_INPUT_CUDA(B);
    CHECK_INPUT_CUDA(out);
    CHECK_DEVICE(A, out);
    CHECK_DEVICE(B, out);

    TVM_FFI_CHECK(A.dtype() == dl_float32, TypeError) << "A must be float32";
    TVM_FFI_CHECK(B.dtype() == dl_float32, TypeError) << "B must be float32";
    TVM_FFI_CHECK(out.dtype() == dl_float32, TypeError) << "out must be float32";

    TVM_FFI_CHECK(A.ndim() == 2, ValueError) << "A must be 2D";
    TVM_FFI_CHECK(B.ndim() == 2, ValueError) << "B must be 2D";
    TVM_FFI_CHECK(out.ndim() == 2, ValueError) << "out must be 2D";

    ffi::CUDADeviceGuard guard(A.device().device_id);
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    // Define the GEMM operation
    using Gemm = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor,
                                             float, cutlass::layout::RowMajor,
                                             float, cutlass::layout::RowMajor>;

    // Create a GEMM object
    Gemm gemm_op;

    // Define the problem size
    cutlass::gemm::GemmCoord problem_size(A.size(0), B.size(1), A.size(1));

    // Define the arguments for the GEMM operation
    typename Gemm::Arguments args(
        problem_size,
        {static_cast<float*>(A.data_ptr()), static_cast<int>(A.size(1))},
        {static_cast<float*>(B.data_ptr()), static_cast<int>(B.size(1))},
        {static_cast<float*>(out.data_ptr()), static_cast<int>(out.size(1))},
        {static_cast<float*>(out.data_ptr()), static_cast<int>(out.size(1))},
        {1.0f, 0.0f}
    );

    // Launch the GEMM operation
    cutlass::Status status = gemm_op(args, nullptr, stream);

    // Check for errors
    if (status != cutlass::Status::kSuccess) {
        TVM_FFI_THROW(RuntimeError) << "CUTLASS GEMM operation failed: "
                                    << cutlassGetStatusString(status);
    }
}