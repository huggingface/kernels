#include <metal_stdlib>
#include "common.h"
using namespace metal;

kernel void relu_forward_kernel_float(device const float *inA [[buffer(0)]],
                                device float *outC [[buffer(1)]],
                                uint index [[thread_position_in_grid]]) {
    // Explicitly write to output
    outC[index] = max(RELU_THRESHOLD, inA[index]);
}

kernel void relu_forward_kernel_half(device const half *inA [[buffer(0)]],
                                device half *outC [[buffer(1)]],
                                uint index [[thread_position_in_grid]]) {
    // Explicitly write to output
    outC[index] = max(static_cast<half>(0.0), inA[index]);
}