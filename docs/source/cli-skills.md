### kernels skills add

Use `kernels skills add` to install the skills for AI coding assistants like Claude, Codex, and OpenCode. For now, only the `cuda-kernels` skill is supported. Skill files are downloaded from the `huggingface/kernels` directory in this [repository](https://github.com/huggingface/kernels/tree/main/skills).

Skills instruct agents how to deal with hardware-specific optimizations, integrate with libraries like diffusers and transformers, and benchmark kernel performance in consistent ways.

Examples:

```bash
# install for Claude in the current project
kernels skills add --claude

# install globally for Codex
kernels skills add --codex --global

# install for multiple assistants
kernels skills add --claude --codex --opencode

# install to a custom destination and overwrite if already present
kernels skills add --dest ~/my-skills --force
### Create a new kernel project
kernels init my-username/my-kernel --skills ~/my-skills
```