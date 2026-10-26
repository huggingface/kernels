# Standalone AOT ReLU

A minimal CPU PyTorch extension: `setup.py` compiles the C++ kernel and `bindings.cpp` exposes it as `upstream_relu._C`. The port replaces this binding with kernel-builder registration; the kernel computation is unchanged.

From the repository root, using a fresh virtual environment with Python 3.10+ and a C++ compiler:

```sh
python -m pip install torch setuptools wheel ninja
MAX_JOBS=2 python -m pip install --no-build-isolation ./kernel-port/tests/examples/relu-aot/upstream
python - <<'PY'
import torch
from upstream_relu import relu

for size in (1, 1023, 1024, 1025):
    x = torch.linspace(-1, 1, size, dtype=torch.float32)
    torch.testing.assert_close(relu(x), torch.relu(x))
print("AOT upstream passed")
PY
```

Use a separate environment for the JIT example: both expose `upstream_relu`. See the [port instructions](../../README.md) to build and run the ported version.
