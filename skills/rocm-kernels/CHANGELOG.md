# Changelog

## v0.2 (2026-03-12)

### Added
- **Transformers integration**: `references/transformers-integration.md` — LLaMA/Mistral/Qwen RMSNorm patching, Flash Attention 2, epsilon handling differences
- **Transformers injection script**: `scripts/transformers_injection_example.py` — minimal runnable example (~150 lines)
- **HuggingFace Kernels Hub integration**: `references/huggingface-kernels-integration.md` — `get_kernel`, `has_kernel`, publishing, ROCm compatibility notes
- **HuggingFace Kernels example script**: `scripts/huggingface_kernels_example.py` — Hub loading, benchmarking, model integration with fallback
- **GEMM template with XCD swizzle**: Template 5 in `kernel-templates.md` — full GEMM kernel with XCD swizzle for MI355X, L2 cache grouping, autotune configs, Python API, and benchmark
- **CHANGELOG.md**: Version tracking for skill iterations

### Fixed
- Broken cross-references: "Template 2" for GEMM → corrected to "Template 5" in `troubleshooting.md`, `kernelbench-classification.md`, and `skill-evaluation-methodology.md`
- R9700 Memory Bandwidth: filled in ~608 GB/s (was TBD) in SKILL.md

### Updated
- `SKILL.md` See Also section: added new integration guides, scripts, and Hub links
- `SKILL.md` argument-hint: added gemm, transformers, huggingface-kernels, get_kernel
- `manifest.txt`: added all new files

## v0.1 (2026-03-10)

### Added
- Initial skill with SKILL.md, 4 kernel templates (RMSNorm, RoPE 3D, GEGLU, AdaLN)
- MI355X and R9700 GPU optimization guides
- Diffusers integration guide (LTX-Video)
- Troubleshooting guide (14 ROCm-specific issues)
- Benchmark scripts: micro-benchmark (`benchmark_kernels.py`) and E2E (`benchmark_e2e.py`)
- LTX-Video injection example (`ltx_kernel_injection_example.py`)
- KernelBench classification and evaluation methodology docs
- Kernel-agent knowledge base
