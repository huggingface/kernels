#include <metal_stdlib>
using namespace metal;

kernel void __KERNEL_NAME_NORMALIZED___forward_kernel_float(device const float *input [[buffer(0)]],
                                device float *output [[buffer(1)]],
                                uint index [[thread_position_in_grid]]) {
    output[index] = input[index] + 1.0f;
}

kernel void __KERNEL_NAME_NORMALIZED___forward_kernel_half(device const half *input [[buffer(0)]],
                                device half *output [[buffer(1)]],
                                uint index [[thread_position_in_grid]]) {
    output[index] = input[index] + static_cast<half>(1.0);
}
