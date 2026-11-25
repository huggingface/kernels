#include <cuda_runtime.h>

#include "utils.h"

__host__ __device__ static long cdiv_impl(long a, long b) {
  return (a + b - 1) / b;
}

__global__ static void cdiv_kernel(long a, long b, long *result) {
  *result = cdiv_impl(a, b);
}

extern "C" {

const char *error_string(int error) {
  return cudaGetErrorString(static_cast<cudaError_t>(error));
}

int runtime_version(int *version) { return cudaRuntimeGetVersion(version); }

int device_count(int *count) {
  cudaError_t error = cudaGetDeviceCount(count);
  // No GPU or driver: report zero devices instead of failing.
  if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver) {
    *count = 0;
    return cudaSuccess;
  }
  return error;
}

long cdiv(long a, long b) { return cdiv_impl(a, b); }

long round_up(long x, long multiple) { return cdiv_impl(x, multiple) * multiple; }

int cdiv_on_device(long a, long b, long *result) {
  long *device_result = nullptr;
  cudaError_t error = cudaMalloc(&device_result, sizeof(long));
  if (error != cudaSuccess) {
    return error;
  }
  cdiv_kernel<<<1, 1>>>(a, b, device_result);
  error = cudaGetLastError();
  if (error == cudaSuccess) {
    error = cudaMemcpy(result, device_result, sizeof(long),
                       cudaMemcpyDeviceToHost);
  }
  cudaFree(device_result);
  return error;
}

}
