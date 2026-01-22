#include <cutlass/gemm/device/gemm.h>
#include <torch/all.h>

void cutlass_gemm(torch::Tensor &out, torch::Tensor const &A, torch::Tensor const &B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");  
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");  
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");  

    TORCH_CHECK(A.is_contiguous(), "A must be a contiguous");  
    TORCH_CHECK(B.is_contiguous(), "B must be a contiguous");  
    TORCH_CHECK(out.is_contiguous(), "out must be a contiguous");  

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
        {A.data_ptr<float>(), A.size(1)},
        {B.data_ptr<float>(), B.size(1)},
        {out.data_ptr<float>(), out.size(1)},
        {out.data_ptr<float>(), out.size(1)},
        {1.0f, 0.0f}
    );

    // Launch the GEMM operation
    cutlass::Status status = gemm_op(args);

    // Check for errors
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM operation failed");
    }
}


