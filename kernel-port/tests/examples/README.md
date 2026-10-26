# Kernel port examples

Two small ReLU ports exercise the CLI from an upstream source tree to the kernel-builder layout:

| Example | Flavor | Porting steps |
| --- | --- | --- |
| [relu-aot](relu-aot/port.kdl) | AOT, native CPU C++ | Prune packaging, move sources, adapt extension imports, add Torch bindings, generate `[torch]` and `[kernel.relu_cpu]`. |
| [relu-jit](relu-jit/port.kdl) | JIT, Triton | Prune packaging, move the package, convert and relativize nested imports, fill in `__init__.py`, generate `[torch-noarch]`. |

Run either from the repository root (Rust is the only requirement for these commands):

```sh
port_output=$(mktemp -d)
cargo run -p kernel-port -- kernel-port/tests/examples/relu-aot/port.kdl \
    --dir kernel-port/tests/examples/relu-aot/upstream --out "$port_output/aot"
cargo run -p kernel-port -- kernel-port/tests/examples/relu-jit/port.kdl \
    --dir kernel-port/tests/examples/relu-jit/upstream --out "$port_output/jit"
```

Both upstreams have minimal standalone packaging and build/run commands: [AOT](relu-aot/upstream/README.md) and [JIT](relu-jit/upstream/README.md). The recipes discard that packaging, adapt imports and bindings, and preserve the kernel computation. Porting itself needs no downloads or GPU.

## Build and run the ported kernels

After running the port commands above, use kernel-builder's local build path. In the AOT example's Python environment, install `kernels`, `cmake`, and `ninja`, then run from the repository root:

```sh
python -m pip install ./kernels cmake ninja
cargo run -p hf-kernel-builder -- create-pyproject "$port_output/aot"
(cd "$port_output/aot" && CMAKE_ARGS="-DGPU_LANG=CPU" MAX_JOBS=2 python setup.py build_kernel)
```

For JIT, use the GPU environment described in its upstream README:

```sh
python -m pip install ./kernels setuptools
cargo run -p hf-kernel-builder -- create-pyproject "$port_output/jit"
(cd "$port_output/jit" && python setup.py build_kernel --backends=cuda)
```

Run this check in the matching environment. For JIT, replace `"$port_output/aot/build" cpu` with `"$port_output/jit/build" cuda`:

```sh
python - "$port_output/aot/build" cpu <<'PY'
import sys
from pathlib import Path

import kernels
import torch

kernel = kernels.get_local_kernel(Path(sys.argv[1]), backend=sys.argv[2])
for size in (1, 1023, 1024, 1025):
    x = torch.linspace(-1, 1, size, dtype=torch.float32, device=sys.argv[2])
    torch.testing.assert_close(kernel.relu(x), torch.relu(x))
print("Ported kernel passed")
PY
```

These are local development builds; no Nix configuration is needed.

## Regression tests

```sh
cargo test -p kernel-port --test port_e2e
```

Two small integration tests run in the existing Rust CI job via `cargo test -p kernel-port`. Each copies upstream into a temporary directory, invokes the compiled CLI from a different working directory, and checks the complete output file set and contents. They also verify that upstream stays unchanged.

Each `expected/` contains the complete ported source tree, including the kernel implementation and, for AOT, the Torch binding. The tests compare directly against these files, so missing, extra, or changed files fail. Only `port-provenance.json` is excluded from the snapshot comparison; its presence is checked separately. Review changes to these snapshots alongside changes to recipes and inputs.

These tests cover porting through the filesystem output; they do not compile or execute kernels and require neither PyTorch nor a GPU. The existing [`relu`](../../../examples/kernels/relu) and [`relu-triton`](../../../examples/kernels/relu-triton) examples separately cover building and executing kernels on supported hardware.
