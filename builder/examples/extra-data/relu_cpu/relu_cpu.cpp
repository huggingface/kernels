#include <torch/all.h>

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __SSE__
void relu_forward_sse(float* out, const float* input, size_t size) {
    size_t i = 0;

    for (; i + 4 <= size; i += 4) {
        __m128 vec_input = _mm_load_ps(input + i);
        __m128 vec_zero = _mm_setzero_ps();
        __m128 vec_output = _mm_max_ps(vec_input, vec_zero);
        _mm_store_ps(out + i, vec_output);
    }

    for (; i < size; ++i) {
        out[i] = input[i] > 0 ? input[i] : 0;
    }
}
#endif

#ifdef __ARM_NEON
void relu_forward_neon(float* out, const float* input, size_t size) {
    size_t i = 0;

    for (; i + 4 <= size; i += 4) {
        float32x4_t vec_input = vld1q_f32(input + i);
        float32x4_t vec_output = vmaxq_f32(vec_input, vdupq_n_f32(0));
        vst1q_f32(out + i, vec_output);
    }

    for (; i < size; ++i) {
        out[i] = input[i] > 0 ? input[i] : 0;
    }
}
#endif

void relu(torch::Tensor &out, torch::Tensor const &input) {
    TORCH_CHECK(out.dtype() == torch::kFloat32, "Output tensor must be of dtype float");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be of dtype float");
    TORCH_CHECK(out.numel() == input.numel(), "Input and output tensors must have the same number of elements");

#if defined(__SSE__)
    relu_forward_sse(out.data_ptr<float>(), input.data_ptr<float>(), input.numel());
#elif defined(__ARM_NEON)
    relu_forward_neon(out.data_ptr<float>(), input.data_ptr<float>(), input.numel());
#else
    #error "Unsupported architecture; please use a CPU with SSE or ARM NEON support."
#endif
}
