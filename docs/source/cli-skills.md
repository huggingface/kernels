### kernel-builder skills add

Use `kernel-builder skills add` to install the skills for AI coding assistants like Claude, Codex, and OpenCode.
Supported skills include:
- `cuda-kernels` (default)
- `rocm-kernels`

Skill files are downloaded from the `huggingface/kernels` directory in this [repository](https://github.com/huggingface/kernels/tree/main/kernel-builder/skills).

Skills instruct agents how to deal with hardware-specific optimizations, integrate with libraries like diffusers and transformers, and benchmark kernel performance in consistent ways.

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
