# Torch stable ABI shim

A header-only shim that makes newer PyTorch stable C++ API wrappers available
when building against older PyTorch stable headers. Wrappers introduced in later
headers can use the stable dispatcher API already available in older versions.
APIs requiring a newer runtime are guarded by `TORCH_FEATURE_VERSION` and are
only exposed when the targeted stable ABI version supports them.

The headers are vendored from PyTorch, with names adjusted to avoid collisions
with PyTorch. Tensor and device types remain in `torch::stable`.

## Copyright and license

See [LICENSE](LICENSE) for the unmodified upstream PyTorch license, including
all copyright notices, conditions, and the disclaimer.
