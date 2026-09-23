# Standalone JIT ReLU

A minimal Python package containing a Triton kernel. Installation packages the Python sources; Triton compiles the kernel on its first call.

From the repository root, using a fresh virtual environment with Python 3.10+, CUDA-enabled PyTorch, Triton, and a supported NVIDIA GPU:

```sh
python -m pip install ./kernel-port/tests/examples/relu-jit/upstream
python - <<'PY'
import torch
from upstream_relu import relu

for size in (1, 1023, 1024, 1025):
    x = torch.linspace(-1, 1, size, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(relu(x), torch.relu(x))
print("JIT upstream passed")
PY
```

Use a separate environment for the AOT example: both expose `upstream_relu`. See the [port instructions](../../README.md) to build and run the ported version.
