# Install agent skills

Use `kernel-builder skills add` to install the skills for AI coding assistants like Claude, Codex, and OpenCode.
Supported skills include:
- `cuda-kernels` (default)
- `rocm-kernels`
- `xpu-kernels`
- `cpu-kernels`

Skill files are downloaded from the `huggingface/kernels` directory in this [repository](https://github.com/huggingface/kernels/tree/main/kernel-builder/skills).

Skills instruct agents how to deal with hardware-specific optimizations, integrate with libraries like diffusers and transformers, and benchmark kernel performance in consistent ways.

> [!TIP]
> **When are CPU kernels actually helpful?** Two main cases:
> - **Better performance on Intel Xeon** — custom AVX2/AVX512 kernels (and AMX via brgemm for quantized GEMM) outperform generic PyTorch ops for element-wise and quantized workloads, especially in CPU-only or latency-sensitive serving.
> - **Enabling functionality that otherwise can't run** — some kernels are a hard requirement, e.g. `megablocks` MoE on CPU, where without the kernel you simply cannot run MXFP4.

Example CPU kernels built with this skill (available on the Hub under [`kernels-community`](https://huggingface.co/kernels-community)):

- [`kernels-community/megablocks`](https://huggingface.co/kernels-community/megablocks) — MoE kernels with a CPU backend that enable running MXFP4 MoE models on CPU.
- [`kernels-community/quantization-gptq`](https://huggingface.co/kernels-community/quantization-gptq) — INT4 quantized GEMM using AVX512.
- [`kernels-community/rmsnorm`](https://huggingface.co/kernels-community/rmsnorm) — RMSNorm with AVX2/AVX512 element-wise paths.

Examples:

```bash
# install for Claude in the current project
kernel-builder skills add --claude

# install ROCm kernels skill for Codex
kernel-builder skills add --skill rocm-kernels --codex

# install globally for Codex
kernel-builder skills add --codex --global

# install for multiple assistants
kernel-builder skills add --claude --codex --opencode

# install to a custom destination and overwrite if already present
kernel-builder skills add --dest ~/my-skills --force
```
