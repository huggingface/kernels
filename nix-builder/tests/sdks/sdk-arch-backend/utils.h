#pragma once

// CUDA side of the extension (`utils.cu`). Functions that can fail return a
// CUDA error code (0 on success), `error_string` returns its description.

#ifdef __cplusplus
extern "C" {
#endif

const char *error_string(int error);

// Version of the CUDA runtime that the extension is linked against, e.g.
// 12060 for CUDA 12.6. Does not need a GPU.
int runtime_version(int *version);

int device_count(int *count);

// Ceiling division and rounding up, for `a >= 0` and `b > 0`. These are
// `__host__ __device__` functions that are called on the CPU here.
long cdiv(long a, long b);
long round_up(long x, long multiple);

int cdiv_on_device(long a, long b, long *result);

#ifdef __cplusplus
}
#endif
