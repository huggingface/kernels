#include <torch/all.h>

// NCCL is a dependency of Torch, so the NCCL headers are available to every
// CUDA kernel without declaring a dependency. Kernels are built against the
// NCCL version of the Torch version that they are built for.
#include <nccl.h>

// The device API headers are only available since NCCL 2.28.
#if __has_include(<nccl_device.h>)
#include <nccl_device.h>
#endif

// 2.27.5 is the lowest NCCL version that we build against (Torch 2.10 with
// CUDA 12).
static_assert(NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 5),
              "NCCL headers are older than expected.");

// The NCCL version that this kernel was compiled against.
int64_t nccl_header_version() { return NCCL_VERSION_CODE; }

// The version of the NCCL library that is loaded at runtime. This calls into
// libnccl, so the kernel must link against it.
int64_t nccl_runtime_version() {
  int version = 0;
  TORCH_CHECK(ncclGetVersion(&version) == ncclSuccess,
              "ncclGetVersion failed");
  return version;
}
