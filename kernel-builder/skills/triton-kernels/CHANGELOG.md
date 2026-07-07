# Changelog

## v1.0.0 (2026-06-24)

- Initial release of vendor-neutral Triton skill
- SKILL.md with core DSL patterns, autotune, numerics, benchmarking, and integration guidance
- Reference kernels: softmax, matmul with tiling, RMSNorm
- references/: autotune guide, kernel patterns (dropout, fused add+norm, SwiGLU, group-major), benchmarking guide
- scripts/: benchmark and correctness test templates
- examples/: fused softmax with correctness test + benchmark harness
