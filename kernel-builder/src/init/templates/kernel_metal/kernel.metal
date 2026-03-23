#include <metal_stdlib>
using namespace metal;

kernel void {{ kernel_name_normalized }}_kernel(device const float *input [[buffer(0)]],
                                              device float *output [[buffer(1)]],
                                              uint index [[thread_position_in_grid]]) {
    output[index] = input[index] + 1.0f;
}
