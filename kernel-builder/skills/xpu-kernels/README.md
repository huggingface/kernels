# XPU Kernels Skill

This skill was adapted from [Xe-Forge](https://github.com/IntelLabs/Xe-Forge) — an LLM-driven optimization framework that transforms PyTorch code into fast Triton kernels for Intel XPU GPUs.

The skill includes Xe-Forge's CLI tools (`scripts/`), knowledge base (`references/`), and the optimization workflow, all integrated into the hf-kernels skill format.

## Full Experience

For the complete Xe-Forge setup — including the ai-bench harness, test kernels, GEMM/reduction templates, annotated examples, and VTune profiling — clone the full project:

```bash
# Clone the repository
git clone https://github.com/IntelLabs/Xe-Forge
cd Xe-Forge

# Install for Intel XPU
uv sync --extra intel
```

## Prerequisites

- Python 3.10+
- PyTorch with XPU support
- [Intel XPU Backend for Triton](https://github.com/intel/intel-xpu-backend-for-triton)
- Intel XPU hardware (tested on Battlemage G21 / Arc Pro B50)
- Intel VTune Profiler 2025+ *(optional — set `vtune_enabled: false` in `scripts/config.yaml` to skip)*

## Install Dependencies

```bash
pip install -r scripts/requirements.txt
```
