#!/usr/bin/env python3
"""
End-to-end benchmark: LTX-Video pipeline with/without custom Triton kernels on ROCm.

Measures total generation time, per-step latency, and peak memory.

Requirements:
    python -m pip install -r skills/rocm-kernels/scripts/requirements.txt

Usage:
    # Baseline (no custom kernels)
    python benchmark_e2e.py --mode baseline

    # With custom Triton kernels
    python benchmark_e2e.py --mode triton

    # With torch.compile
    python benchmark_e2e.py --mode compile

    # Compare all three
    python benchmark_e2e.py --mode all

    # Quick test (fewer frames/steps)
    python benchmark_e2e.py --mode triton --num-frames 9 --steps 5

    # Save videos and JSON to a structured directory
    python benchmark_e2e.py --mode all \
        --output-dir examples/ltx-video-benchmark \
        --output-json examples/ltx-video-benchmark/results.json
"""
import os
os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'

import argparse
import json
import time

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ============================================================================
# Triton RMSNorm Kernel (same as in benchmark_kernels.py)
# ============================================================================

@triton.jit
def _rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr,
    stride_x_row, D,
    eps,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(x_ptr + row * stride_x_row + offs, mask=mask, other=0.0).to(tl.float32)
    sq_sum = tl.sum(x * x, axis=0)
    rms_inv = tl.rsqrt(sq_sum / D + eps)

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        out = x * rms_inv * w
    else:
        out = x * rms_inv

    tl.store(out_ptr + row * stride_x_row + offs, out.to(x.dtype), mask=mask)


def triton_rmsnorm(x, weight=None, eps=1e-6):
    x_flat = x.contiguous().view(-1, x.shape[-1])
    out = torch.empty_like(x_flat)
    M, D = x_flat.shape
    has_weight = weight is not None
    if not has_weight:
        weight = torch.ones(D, device=x.device, dtype=x.dtype)
    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    _rmsnorm_kernel[(M,)](
        x_flat, weight, out, x_flat.stride(0), D, float(eps), has_weight,
        BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=2,
    )
    return out.view_as(x)


# ============================================================================
# Attention Processor (uses Triton RMSNorm)
# ============================================================================

class TritonLTXVideoAttnProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        from diffusers.models.transformers.transformer_ltx import apply_rotary_emb
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = triton_rmsnorm(query, attn.norm_q.weight, eps=attn.norm_q.eps)
        key = triton_rmsnorm(key, attn.norm_k.weight, eps=attn.norm_k.eps)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# ============================================================================
# Module Patchers
# ============================================================================

def patch_rmsnorm_modules(model):
    patched = 0
    for name, module in model.named_modules():
        if type(module).__name__ == 'RMSNorm':
            eps = getattr(module, 'eps', 1e-6)
            has_weight = hasattr(module, 'weight') and module.weight is not None
            if has_weight:
                def make_fwd(mod, e):
                    def fwd(x): return triton_rmsnorm(x, mod.weight, eps=e)
                    return fwd
                module.forward = make_fwd(module, eps)
            else:
                def make_fwd_nw(e):
                    def fwd(x): return triton_rmsnorm(x, None, eps=e)
                    return fwd
                module.forward = make_fwd_nw(eps)
            patched += 1
    return patched


def inject_triton_kernels(pipe):
    stats = {'attention_processors': 0, 'rmsnorm_modules': 0}
    if not hasattr(pipe, 'transformer'):
        return stats
    for name, module in pipe.transformer.named_modules():
        if hasattr(module, 'set_processor') and hasattr(module, 'processor'):
            module.set_processor(TritonLTXVideoAttnProcessor())
            stats['attention_processors'] += 1
    stats['rmsnorm_modules'] = patch_rmsnorm_modules(pipe.transformer)
    return stats


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(mode, prompt, num_frames, height, width, steps,
                  guidance_scale, seed, warmup_iters, output_dir):
    from diffusers import LTXPipeline
    from diffusers.utils import export_to_video

    device = "cuda"
    dtype = torch.bfloat16

    print(f"\n{'='*60}")
    print(f"MODE: {mode.upper()}")
    print(f"{'='*60}")

    pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=dtype)
    pipe.to(device)

    if mode == "triton":
        stats = inject_triton_kernels(pipe)
        print(f"  Attention processors: {stats['attention_processors']}")
        print(f"  RMSNorm patched: {stats['rmsnorm_modules']}")
    elif mode == "compile":
        pipe.transformer.compile_repeated_blocks(fullgraph=True)
        print("  torch.compile enabled (fullgraph=True)")
    else:
        print("  Baseline (no optimization)")

    # Warmup
    if warmup_iters > 0:
        print(f"\n  Warmup ({warmup_iters} iters, {min(steps, 5)} steps)...")
        for i in range(warmup_iters):
            _ = pipe(prompt=prompt, num_frames=num_frames, height=height, width=width,
                     num_inference_steps=min(steps, 5), guidance_scale=guidance_scale)
            torch.cuda.synchronize()

    # Benchmark run
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"\n  Generating ({num_frames} frames, {steps} steps)...")
    torch.cuda.synchronize()
    start = time.time()
    output = pipe(
        prompt=prompt, num_frames=num_frames, height=height, width=width,
        num_inference_steps=steps, guidance_scale=guidance_scale,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    torch.cuda.synchronize()
    gen_time = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    result = {
        'mode': mode,
        'gen_time_s': round(gen_time, 2),
        'time_per_frame_s': round(gen_time / num_frames, 3),
        'time_per_step_s': round(gen_time / steps, 3),
        'peak_memory_gb': round(peak_mem, 2),
    }

    print(f"\n  Results:")
    print(f"    Total:      {result['gen_time_s']:.2f} s")
    print(f"    Per frame:  {result['time_per_frame_s']:.3f} s")
    print(f"    Per step:   {result['time_per_step_s']:.3f} s")
    print(f"    Peak mem:   {result['peak_memory_gb']:.2f} GB")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"ltx_video_{mode}.mp4")
    export_to_video(output.frames[0], out_path, fps=24)
    print(f"    Saved to:   {out_path}")

    del pipe
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="E2E LTX-Video benchmark on ROCm")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["baseline", "triton", "compile", "all"])
    parser.add_argument("--prompt", type=str,
                        default="A cat sleeping in warm sunlight, cinematic, 4K")
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for generated output files")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON for comparison")
    args = parser.parse_args()

    print("=" * 60)
    print("LTX-Video End-to-End Benchmark (ROCm)")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"ROCm:   {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
    print(f"Config: {args.num_frames} frames, {args.height}x{args.width}, {args.steps} steps")

    modes = ["baseline", "triton", "compile"] if args.mode == "all" else [args.mode]
    all_results = []

    for mode in modes:
        r = run_benchmark(mode, args.prompt, args.num_frames, args.height,
                          args.width, args.steps, args.guidance_scale,
                          args.seed, args.warmup, args.output_dir)
        all_results.append(r)

    # Comparison table
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Mode':<12} {'Time (s)':<12} {'Per Step (s)':<15} {'Peak Mem (GB)':<15}")
        print("-" * 54)
        baseline_time = all_results[0]['gen_time_s']
        for r in all_results:
            speedup = baseline_time / r['gen_time_s'] if r['gen_time_s'] > 0 else 0
            suffix = f" ({speedup:.2f}x)" if r['mode'] != 'baseline' else ""
            print(f"{r['mode']:<12} {r['gen_time_s']:<12.2f} {r['time_per_step_s']:<15.3f} {r['peak_memory_gb']:<15.2f}{suffix}")

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
