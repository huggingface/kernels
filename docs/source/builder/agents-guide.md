# Writing custom kernels with code agents

Code agents are a good fit to build custom kernels because the hard part is not just writing in Domain Specific Language (DSLs) like CUDA. You also need the right project layout, PyTorch bindings, architecture-specific choices, model-specific integration, and trustworthy benchmarks. 

Kernels on Hugging Face are compatible with agents via skills and the `hf` CLI. The `cuda-kernels` and `rocm-kernels` skills contain knowledge so an agent can generate and publish a complete kernel project, instead of isolated snippets.

This guide is for **authoring new kernels**. If you only want to **load an existing precompiled kernel**, use `get_kernel()` instead.

## Before you start

You need:

- a coding agent that supports skills, such as Claude Code, Codex, Cursor, or OpenCode  
- a clear target: library, model, operation, GPU, dtype, and representative shapes

The skill currently focuses on NVIDIA GPUs such as **H100**, **A100**, and **T4**, and on integration patterns for **transformers** and **diffusers**.

Install the skill into your agent. If you need the latest version from `main`, use:

```shell
cargo install --git https://github.com/huggingface/kernels hf-kernel-builder

# Install your skills. Use --claude, --codex, or --opencode
kernel-builder skills add --claude
```

> [!NOTE]
> Check [this example](https://github.com/burtenshaw/kernel-skill/tree/main/examples/ltx_video) to see what generated kernels look like.

## 1. Give the agent a precise task prompt

Writing kernels is a hard problem, so be specific to agents. A robust prompt will declare all core attributes, including:

- the library, for example `transformers` or `diffusers`  
- the model id, for example `Qwen3-8B` or `LTX-Video`  
- the operation, for example `RMSNorm`, attention, RoPE, `GEGLU`, or `AdaLN`  
- the target GPU, for example `H100`, `A100`, or `T4`  
- the dtype, for example `bfloat16`, `float16`, or `float32`  
- the outputs you expect: kernel code, bindings, tests, and benchmarks

In practice, you can often skip some of these and the agent will infer based on common practice, but if you know a detail declare it. 

For example:

```
Build a vectorized RMSNorm kernel for H100 targeting Qwen3-8B in transformers.
Create the full kernel-builder project, PyTorch bindings, correctness tests, and benchmark scripts.
```

Or for diffusers:

```
Build an H100 RMSNorm kernel for LTX-Video in diffusers.
Patch the pipeline correctly, benchmark it against the PyTorch baseline, and report end-to-end impact.
```

If you prefer, you can first scaffold a project with `kernel-builder init --name <org>/<kernel>` and then ask the agent to fill in the implementation.

## 2. Verify that the agent produces a complete kernel project

A useful result is a full `kernel-builder` project, not just a `.cu` file. The exact layout can vary, but it should include at least:

```
examples/your_model/
├── kernel_src/
│   └── rmsnorm.cu              # Vectorized CUDA kernel
├── torch-ext/
│   ├── your_kernels/__init__.py
│   └── torch_binding.cpp       # PyTorch C++ bindings
├── benchmark_rmsnorm.py        # Micro-benchmark script
├── build.toml                  # kernel-builder config
├── setup.py                    # pip install -e .
└── pyproject.toml
```

The agent skills contain example scipts to help you verify the project. So you can briefly test it yourself by running:

```
Verify the kernel project works with a transformers example.
```

## 3. Review the generated files

Let's dive deeper into the generated files, and explore how to validate the project.

### `build.toml`

This is the main configuration file for `kernel-builder`. It tells `kernel-builder` what to build and how so it should contain all the core information about your kernel project.

```
[general]
name = "your_kernels"
backends = ["cuda"]
version = 1

[torch]
src = ["torch-ext/torch_binding.cpp"]

[kernel.rmsnorm]
backend = "cuda"
src = ["kernel_src/rmsnorm.cu"]
depends = ["torch"]
cuda-capabilities = ["9.0"]  # H100
```

First check that:

- `backends = ["cuda"]` is correct for your project  
- the kernel source files are listed correctly  
- the Torch binding sources are included under `[torch]`  
- `cuda-capabilities` is only set when the kernel truly targets specific architectures

For architecture-specific kernels, typical capability values are:

- H100: `9.0`  
- A100: `8.0`  
- T4: `7.5`

If the kernel does **not** require a specific capability, the kernels docs recommend leaving `cuda-capabilities` unset so the builder can target all supported capabilities. In practice, you can prompt your agent to review the `build.toml` for excessive definitions. Agents have a tendency to over-specify capabilities.

### Torch bindings

The kernel should be registered as Torch ops in `torch-ext/torch_binding.cpp`, with declarations in a header and a small Python wrapper in `torch-ext/<name>/__init__.py`. This is what makes the kernel callable from Python and is the right foundation for `torch.compile` compatibility.

### Model integration code

Make sure the integration matches the library:

- **transformers**: patch the target modules directly, often RMSNorm modules whose class name contains `RMSNorm`  
- **diffusers**: inspect the actual pipeline structure before patching, because modules and attention processors can differ across pipelines

> [!NOTE]
> One common issue is that the agent will not integrate the kernel at all. Typically because the project's context is so long.

A few patterns matter in practice for the integration code:

- In **transformers**, RMSNorm modules generally have weights, but epsilon may be exposed as `variance_epsilon` or `eps` depending on the model.  
- In **diffusers**, some RMSNorm modules may have `weight=None`, so the integration code needs to handle both weighted and unweighted cases.  
- In **diffusers**, checking `type(module).__name__` is often more reliable than `isinstance(...)` for matching RMSNorm modules across implementations.  
- If a diffusers pipeline uses CPU offloading, inject custom kernels **before** enabling offload.

For attention, prefer the model library's existing optimized path when one already exists. For example, in `transformers`, Flash Attention 2 is usually the right baseline for attention, while custom kernels are especially useful for operations like RMSNorm and other targeted hotspots.

## 5. Build and test, and benchmark

Kernel Hub kernels must support all recent PyTorch and CUDA configurations. The kernel-builder Nix flake handles this automatically. Copy the [example `flake.nix`](https://github.com/huggingface/kernels/blob/main/builder/examples/relu/flake.nix) into your project and run:

```shell
nix flake update
nix run .#build-and-copy -L
```

This builds the kernel for every required PyTorch/CUDA variant and places the results in `build/`. For faster builds, enable the HuggingFace Nix cache:

```shell
nix run nixpkgs#cachix -- use huggingface
```

## 6. Benchmark

There are two main benchmarks to consider:

1. an isolated kernel micro-benchmark  
2. an end-to-end benchmark in the real model or pipeline

The agent will generate both benchmarks based on the agent skills examples. Typically as a script called `benchmark_example.py`. If you have access to the target hardware, you can run it to verify the kernel works. For example, the agent will generat a table like this:

```markdown
| Shape | Custom (ms) | PyTorch (ms) | Speedup |
| :---- | :---: | :---: | :---: |
| [1x128x4096] | 0.040 | 0.062 | **1.58x** |
| [1x512x4096] | 0.038 | 0.064 | **1.69x** |
| [1x1024x4096] | 0.037 | 0.071 | **1.90x** |
| [1x2048x4096] | 0.045 | 0.091 | **2.03x** |
| [1x4096x4096] | 0.071 | 0.150 | **2.12x** |
| [4x512x4096] | 0.056 | 0.093 | **1.67x** |
| [8x256x4096] | 0.045 | 0.092 | **2.06x** |
| [1x8192x4096] | 0.109 | 0.269 | **2.47x** |
```

Interpret the results carefully. A kernel can show a large isolated speedup but only a modest end-to-end gain if that operation is a small fraction of total runtime. In the LTX-Video example from [the blog we wrote](https://huggingface.co/blog/custom-cuda-kernels-agent-skills), the generated RMSNorm kernel improved the isolated benchmark by about **1.88x** on average, but end-to-end video generation improved by about **6%**, which matched the fact that RMSNorm accounted for only a small share of total compute.

## 7. Publish to the Hub

Once the project is correct and benchmarked, you can build Hub-compatible artifacts and upload them. For this, you should first push to the Hub using the `hf` CLI tool:

```shell
# install the hf CLI tool
hf skills add 

# Authenticate
hf auth login

# Push to the Hub
<agent-prompt>
Push the kernel to the Hub.
</agent-prompt>
```

Or, you can manually create the repository and upload the artifacts:

```shell
# Create the repository
hf repo create your-org/your-kernel --type model

# Upload the artifacts
# Run inside the main kernel directory, where build/ is.
kernel-builder upload
```

After pushing to the Hub, users can load the kernel without compiling:

```py
from kernels import get_kernel

kernel = get_kernel("your-org/your-kernel", version=1)
```

Well done! You have now built a custom kernel and published it to the Hub.