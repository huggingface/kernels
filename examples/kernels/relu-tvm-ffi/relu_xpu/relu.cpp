#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <sycl/sycl.hpp>

#include "../util.hh"

using namespace sycl;

void relu_xpu_impl(float *output, float const *input, int const numel) {
    // Create SYCL queue directly
    sycl::queue queue;

    // Launch SYCL kernel
    queue.parallel_for(range<1>(numel), [=](id<1> idx) {
        auto i = idx[0];
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }).wait();
}

using namespace tvm;

void relu_xpu(ffi::TensorView out, ffi::TensorView const input) {
    CHECK_INPUT_XPU(input);
    CHECK_INPUT_XPU(out);
    CHECK_DEVICE(input, out);

    TVM_FFI_CHECK(input.dtype() == out.dtype(), ValueError) << "input/output dtype mismatch";
    TVM_FFI_CHECK(input.numel() == out.numel(), ValueError) << "input/output size mismatch";

    auto numel = input.numel();

    if (input.dtype() == dl_float32) {
        relu_xpu_impl(static_cast<float *>(out.data_ptr()),
                      static_cast<float const *>(input.data_ptr()),
                      numel);
    } else {
        TVM_FFI_THROW(TypeError) << "Unsupported dtype: " << input.dtype();
    }
}
