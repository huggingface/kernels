### install_skills.py

Use `install_skills.py` (located in `kernel-builder/scripts/`) to install the skills for AI coding assistants like Claude, Codex, and OpenCode. For now, only the `cuda-kernels` skill is supported. Skill files are downloaded from the `huggingface/kernels` directory in this [repository](https://github.com/huggingface/kernels/tree/main/kernel-builder/skills).

Skills instruct agents how to deal with hardware-specific optimizations, integrate with libraries like diffusers and transformers, and benchmark kernel performance in consistent ways.

Examples:

```bash
# install for Claude in the current project
python kernel-builder/scripts/install_skills.py --claude

# install globally for Codex
python kernel-builder/scripts/install_skills.py --codex --global

# install for multiple assistants
python kernel-builder/scripts/install_skills.py --claude --codex --opencode

# install to a custom destination and overwrite if already present
python kernel-builder/scripts/install_skills.py --dest ~/my-skills --force
```