# Implementation Reference

Code templates and patterns for C++ CPU kernel development with AVX2/AVX512 intrinsics.

## Template Selection

Start with a template that matches your kernel type:
- **Element-wise** (RMSNorm, activations): Direct AVX512 intrinsics, vector type abstractions
- **GEMM** (quantized GEMM, MoE): tinygemm + brgemm dual-path, Unroll<N> template
- **Attention** (Flash-Attention): Tiled attention with brgemm for matmul blocks

## Core File Structure

Every CPU kernel follows this structure:

```
my_kernel/
├── my_kernel_cpu/
│   ├── cpu_features.hpp          # CPUID detection (own namespace)
│   ├── my_kernel_cpu.cpp         # Dispatcher
│   ├── my_kernel_cpu.hpp         # Shared declarations
│   ├── my_kernel_cpu_torch.cpp   # Python ↔ C++ bridge
│   ├── my_kernel_avx512.cpp      # AVX512 implementation
│   └── my_kernel_avx512.hpp      # AVX512 declarations
├── torch-ext/
│   └── torch_binding.cpp         # Op registration
└── build.toml                    # Multi-target compilation
```

## cpu_features.hpp (Per-Kernel, Own Namespace)

Each kernel has its OWN copy of `cpu_features.hpp` in its OWN namespace to avoid ODR violations:

```cpp
#pragma once
#include <cpuid.h>

namespace my_kernel_cpu {

class CPUFeatures {
public:
    static bool hasAVX2() {
        unsigned int eax, ebx, ecx, edx;
        if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
        return (ebx >> 5) & 1;  // AVX2 bit
    }

    static bool hasAVX512BF16() {
        unsigned int eax, ebx, ecx, edx;
        // Check AVX512F first
        if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
        if (!((ebx >> 16) & 1)) return false;  // AVX512F

        // Check OS support via XCR0
        unsigned int xcr0_lo, xcr0_hi;
        asm volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
        if ((xcr0_lo & 0xe6) != 0xe6) return false;  // OS support for ZMM

        // Check AVX512_BF16 (CPUID leaf 7, sub-leaf 1)
        if (!__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) return false;
        return (eax >> 5) & 1;  // AVX512_BF16 bit
    }

    // For GEMM kernels that need AMX via brgemm
    static bool hasAMX() {
        unsigned int eax, ebx, ecx, edx;
        if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
        return (edx >> 24) & 1;  // AMX-TILE bit
    }

    // Composite check for kernels requiring multiple features
    static bool hasAllRequiredFeatures() {
        return hasAVX512BF16();  // Add hasAMX() for GEMM kernels
    }
};

}  // namespace my_kernel_cpu
```

## Dispatcher Pattern (my_kernel_cpu.cpp)

Most kernels: AVX512 → ATen fallback (two tiers):

```cpp
#include "cpu_features.hpp"
#include "my_kernel_avx512.hpp"
#include <torch/torch.h>

namespace my_kernel_cpu {

void my_kernel_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    float eps
) {
    if (CPUFeatures::hasAVX512BF16()) {
        avx512::my_kernel_impl(output, input, weight, eps);
    } else {
        // ATen fallback — works on any CPU
        auto variance = input.to(torch::kFloat32).pow(2).mean(-1, true);
        output = input * torch::rsqrt(variance + eps) * weight;
    }
}

}  // namespace my_kernel_cpu
```

## Bridge File (my_kernel_cpu_torch.cpp)

Bridges Python-facing tensor API to internal C++ implementation:

```cpp
#include <torch/torch.h>
#include "my_kernel_cpu.hpp"

torch::Tensor my_kernel_cpu_forward(
    torch::Tensor input,
    torch::Tensor weight,
    float eps
) {
    auto output = torch::empty_like(input);
    my_kernel_cpu::my_kernel_forward(output, input, weight, eps);
    return output;
}
```

## Element-wise Kernel Template (AVX512)

```cpp
#include <immintrin.h>
#include <torch/torch.h>
#include <omp.h>

namespace my_kernel_cpu {
namespace avx512 {

constexpr int VEC_ELEM_NUM = 16;  // fp32 elements per __m512

void rmsnorm_impl(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    float eps
) {
    auto num_tokens = input.size(0);
    auto hidden_size = input.size(1);

    auto input_data = input.data_ptr<float>();
    auto weight_data = weight.data_ptr<float>();
    auto output_data = output.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_tokens; ++i) {
        const float* x = input_data + i * hidden_size;
        float* y = output_data + i * hidden_size;

        // 1. Compute variance (sum of squares)
        __m512 sum_sq = _mm512_setzero_ps();
        int64_t j = 0;
        for (; j + VEC_ELEM_NUM <= hidden_size; j += VEC_ELEM_NUM) {
            __m512 v = _mm512_loadu_ps(x + j);
            sum_sq = _mm512_fmadd_ps(v, v, sum_sq);
        }
        float variance = _mm512_reduce_add_ps(sum_sq);
        // Handle tail
        for (; j < hidden_size; ++j) {
            variance += x[j] * x[j];
        }
        variance /= hidden_size;

        // 2. Compute rsqrt(variance + eps)
        float inv_rms = 1.0f / sqrtf(variance + eps);
        __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);

        // 3. Normalize and scale by weight
        j = 0;
        for (; j + VEC_ELEM_NUM <= hidden_size; j += VEC_ELEM_NUM) {
            __m512 v = _mm512_loadu_ps(x + j);
            __m512 w = _mm512_loadu_ps(weight_data + j);
            __m512 result = _mm512_mul_ps(_mm512_mul_ps(v, inv_rms_vec), w);
            _mm512_storeu_ps(y + j, result);
        }
        for (; j < hidden_size; ++j) {
            y[j] = x[j] * inv_rms * weight_data[j];
        }
    }
}

}  // namespace avx512
}  // namespace my_kernel_cpu
```

## GEMM Kernel Patterns

### Unroll<N> Template (Used in ALL GEMM Kernels)

Compile-time loop unrolling via template recursion. Uses `std::integral_constant` for compile-time index:

```cpp
#define ALWAYS_INLINE __attribute__((always_inline)) inline

template <int n>
struct Unroll {
    template <typename Func, typename... Args>
    ALWAYS_INLINE void operator()(const Func &f, Args... args) const {
        Unroll<n - 1>{}(f, args...);
        f(std::integral_constant<int, n - 1>{}, args...);
    }
};

template <>
struct Unroll<1> {
    template <typename Func, typename... Args>
    ALWAYS_INLINE void operator()(const Func &f, Args... args) const {
        f(std::integral_constant<int, 0>{}, args...);
    }
};

// Usage: Unroll<ROWS * COLS>{}(compute_lambda, k);
// The lambda receives std::integral_constant<int, i> as first arg.
```

### tinygemm_kernel_nn (Small-M GEMM Micro-Kernel)

A struct template with static `apply()` method. For M ≤ 4 with bf16, uses fused dequant + _mm512_dpbf16_ps:

```cpp
template <typename scalar_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
    // Primary template — static_assert triggers for unsupported types
    static inline void apply(...) { static_assert(sizeof(scalar_t) == 0, "unsupported"); }
};

// BFloat16 specialization
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, BLOCK_M, BLOCK_N> {
    static inline void apply(
        const at::BFloat16* __restrict__ A,
        const unsigned char* __restrict__ B,
        at::BFloat16* __restrict__ C,
        const uint8_t* __restrict__ Bz,       // zero-points (GPTQ only)
        const at::BFloat16* __restrict__ Bs,   // scales
        int64_t K, int blocksize,
        int64_t lda, int64_t ldb, int64_t ldc,
        int64_t strideBz, int64_t strideBs
    ) {
        constexpr int COLS = BLOCK_N / 16;
        __m512 vc[BLOCK_M * COLS] = {};  // fp32 accumulators

        // pre_compute: load zeros, LUT, etc.
        auto compute = [&](auto i, int k) {
            // 1. nibble split + zero subtract + LUT lookup → bf16
            // 2. Unroll<BLOCK_M>{}(load_a_and_dpbf16, k);
        };

        auto scale_and_store = [&](auto i) {
            // scale fmadd per group, convert fp32 → bf16, store
        };

        int64_t K2 = K >> 1;  // dpbf16_ps processes 2 bf16 pairs
        for (int64_t k = 0; k < K2; ++k) {
            Unroll<BLOCK_M * COLS>{}(compute, (int)k);
            // scale at group boundaries
        }
        Unroll<BLOCK_M * COLS>{}(scale_and_store);
    }
};
```

### parallel_2d Threading (GEMM Kernels)

Custom 2D thread decomposition for matrix operations. Template function (not std::function):

```cpp
inline int adjust_num_threads(int m) {
    int nth = at::get_num_threads();
    if (m == 1) return 1;
    return std::max(1, (nth >> 1) * 2);  // round to even
}

inline int div_up(int a, int b) { return (a + b - 1) / b; }

template <typename func_t>
inline void parallel_2d(
    int m, int n,
    const func_t& f
) {
    int nth = adjust_num_threads(m);
    // Factor nth into nth_m * nth_n based on M/N ratio
    int nth_m = 1, nth_n = nth;
    while (nth_m < nth && nth_m * 2 <= m) {
        nth_m *= 2;
        nth_n = nth / nth_m;
    }

    #pragma omp parallel num_threads(nth)
    {
        int tid = omp_get_thread_num();
        int tm = tid / nth_n;
        int tn = tid % nth_n;
        int m_start = tm * div_up(m, nth_m);
        int m_end = std::min(m, (tm + 1) * div_up(m, nth_m));
        int n_start = tn * div_up(n, nth_n);
        int n_end = std::min(n, (tn + 1) * div_up(n, nth_n));
        if (m_start < m_end && n_start < n_end) {
            f(m_start, m_end, n_start, n_end);
        }
    }
}
```

### tinygemm vs brgemm Selection

The `tinygemm_kernel` function wraps both paths with `parallel_2d`:

```cpp
template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t *A, const unsigned char *B,
    scalar_t *C, const uint8_t* Bz, const scalar_t *Bs,
    scalar_t *Btmp, float *Ctmp,
    int64_t M, int64_t N, int64_t K, int blocksize,
    int64_t lda, int64_t ldb, int64_t ldc,
    int64_t strideBz, int64_t strideBs,
    bool brg, bool use_brgemm_dequant_out = false
) {
    // brg = (M > 4) — set by caller
    parallel_2d(div_up(M, BLOCK_M), div_up(N, BLOCK_N),
        [&](int mb_start, int mb_end, int nb_start, int nb_end) {
            for (int mb = mb_start; mb < mb_end; ++mb) {
                for (int nb = nb_start; nb < nb_end; ++nb) {
                    if (brg) {
                        // brgemm path: dequant B block → brgemm
                        brgemm<scalar_t>::apply(...);
                    } else {
                        // tinygemm path: fused dequant+GEMM
                        tinygemm_kernel_nn<scalar_t, BLOCK_M, NB_SIZE>::apply(...);
                    }
                }
            }
        }
    );
    if (brg) at::native::cpublas::brgemm_release();
}

// Caller (in gemm_int4_inference):
const bool use_brgemm = M > 4;
const bool use_brgemm_dequant_out = M > 100;  // pre-dequant all B
tinygemm_kernel<scalar_t>(..., use_brgemm, use_brgemm_dequant_out);
```

## torch_binding.cpp Registration

Located at `torch-ext/torch_binding.cpp`:

```cpp
#include "registration.h"

#if defined(CPU_KERNEL)
torch::Tensor my_kernel_cpu_forward(torch::Tensor input, torch::Tensor weight, float eps);
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("forward(Tensor input, Tensor weight, float eps) -> Tensor");
    ops.impl("forward", torch::kCPU, &my_kernel_cpu_forward);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
```

> **Note**: rmsnorm uses `c10::DispatchKey::CompositeExplicitAutograd` instead of `torch::kCPU`.

## Vector Type Abstractions (cpu_types_avx512.hpp)

Used by element-wise kernels (rmsnorm). Not used by GEMM kernels or flash-attn2.

```cpp
struct FP32Vec16 {
    __m512 reg;
    FP32Vec16(float v) : reg(_mm512_set1_ps(v)) {}
    FP32Vec16(__m512 r) : reg(r) {}
    FP32Vec16 operator*(const FP32Vec16& other) const {
        return FP32Vec16(_mm512_mul_ps(reg, other.reg));
    }
    FP32Vec16 operator+(const FP32Vec16& other) const {
        return FP32Vec16(_mm512_add_ps(reg, other.reg));
    }
    float reduce_sum() const { return _mm512_reduce_add_ps(reg); }
};

struct BF16Vec32 {
    __m512i reg;
    BF16Vec32(__m512i r) : reg(r) {}
    // Convert to two FP32Vec16
    void convert(FP32Vec16& lo, FP32Vec16& hi) const {
        __m256i lo_half = _mm512_castsi512_si256(reg);
        __m256i hi_half = _mm512_extracti64x4_epi64(reg, 1);
        lo = FP32Vec16(_mm512_cvtpbh_ps((__m256bh)lo_half));
        hi = FP32Vec16(_mm512_cvtpbh_ps((__m256bh)hi_half));
    }
};
```
