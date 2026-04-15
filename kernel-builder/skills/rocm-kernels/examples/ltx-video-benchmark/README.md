# ROCm Kernels Examples: LTX-Video Benchmark

This is the consolidated reviewer-facing README for ROCm kernels benchmark examples.

It answers:

> Could we also see some numbers with and without these kernels?

## Dependencies

```bash
python -m pip install -r skills/rocm-kernels/scripts/requirements.txt
```

## What is included

- A formal benchmark table comparing `baseline` and `triton`
- A supplemental `compile` run for reference
- A single consolidated JSON file: `benchmark_results.json`
- A single live harness trace from OpenCode

## Benchmark configuration

- Model: `Lightricks/LTX-Video`
- Device: `AMD Radeon Graphics`
- ROCm: `7.1.52802-26aae437f6`
- Config: `25 frames`, `480x704`, `30 steps`, `warmup=1`
- Repeats: `3` for `baseline` and `triton`
- Same prompt and seed across modes

## Commands

Run from `skills/rocm-kernels`:

```bash
# Regenerate benchmark outputs
python scripts/benchmark_e2e.py --mode all --num-frames 25 --steps 30 --warmup 1 --output-dir examples/ltx-video-benchmark
```

## Summary table (formal benchmark)

| Mode | gen_time_s | time_per_step_s | peak_memory_gb | speedup |
|------|-----------:|----------------:|---------------:|--------:|
| baseline (mean of 3) | 6.91 | 0.231 | 18.58 | 1.00x |
| triton (mean of 3) | 6.10 | 0.203 | 18.58 | 1.13x |
| compile (single run) | 5.05 | 0.168 | 18.58 | 1.37x |

Notes:

- `speedup` is computed against baseline mean.
- `compile` is included as a supplemental reference and is not used as the main claim for the custom-kernel comparison.

## Output files

- `benchmark_results.json`
- `trace/opencode_live/results.json`
- `trace/opencode_live/opencode_trace_result.json`

## Coding Harness Trace

This package includes a single coding-harness trace file:

- `trace/opencode_live/opencode_trace_result.json`

OpenCode trace was generated with model `opencode/minimax-m2.5-free`.

## PR-ready response

We added baseline vs Triton numbers for LTX-Video on ROCm with consolidated benchmark output and a single live harness trace. Using the same prompt, seed, resolution, frame count, and inference steps, Triton improves end-to-end latency from `6.91s` to `6.10s` on average (`1.13x` speedup) in the formal benchmark package.
