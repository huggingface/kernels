#include <torch/all.h>

// NCCL is a dependency of Torch, so the NCCL headers (including the device API
// headers) are available to every CUDA kernel without declaring a dependency.
#include <nccl.h>
#include <nccl_device.h>

static_assert(NCCL_VERSION_CODE >= NCCL_VERSION(2, 32, 0),
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
