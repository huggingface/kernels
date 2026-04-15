#!/usr/bin/env python3
"""
Minimal example: Inject custom Triton kernels into HuggingFace Transformers models on ROCm.

This script demonstrates the essential pattern for integrating custom Triton kernels
with transformers models like LLaMA, Mistral, and Qwen on AMD GPUs.

Key lessons:
1. Transformers RMSNorm modules always have weights (unlike some diffusers modules)
2. Use 'RMSNorm' substring match to catch LlamaRMSNorm, MistralRMSNorm, etc.
3. Check for 'variance_epsilon' (LLaMA) or 'eps' (others) for epsilon value
4. Use Flash Attention 2 for attention optimization instead of custom processors
5. ROCm: tl.libdevice/tl.math.tanh NOT available — use manual math

Usage:
    python scripts/transformers_injection_example.py

Requirements:
    python -m pip install -r skills/rocm-kernels/scripts/requirements.txt
"""

import os
import sys
import time

os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'

import torch
import torch.nn as nn
import triton
import triton.language as tl


# =============================================================================
# Triton RMSNorm Kernel (ROCm compatible)
# =============================================================================

@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, weight_ptr, out_ptr,
    stride_x, D, eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / D
    rms_inv = tl.rsqrt(variance + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    out = x * rms_inv * w

    tl.store(out_ptr + row * stride_x + offs, out.to(x.dtype), mask=mask)


def triton_rmsnorm(x, weight, eps=1e-6):
    x_2d = x.contiguous().view(-1, x.shape[-1])
    out = torch.empty_like(x_2d)
    M, D = x_2d.shape
    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    rmsnorm_fwd_kernel[(M,)](
        x_2d, weight, out, x_2d.stride(0), D, float(eps),
        BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=2,
    )
    return out.view_as(x)


# =============================================================================
# RMSNorm Module Patcher
# =============================================================================

def patch_rmsnorm_modules(model: nn.Module) -> int:
    """
    Patch all RMSNorm modules to use Triton kernel on ROCm.

    Works with LlamaRMSNorm, MistralRMSNorm, Qwen2RMSNorm, etc.
    Unlike diffusers, transformers RMSNorm always has weights.
    """
    patched_count = 0

    for name, module in model.named_modules():
        class_name = type(module).__name__

        if 'RMSNorm' in class_name:
            eps = getattr(module, 'variance_epsilon', None)
            if eps is None:
                eps = getattr(module, 'eps', 1e-6)

            has_weight = hasattr(module, 'weight') and module.weight is not None

            if has_weight:
                def make_patched_forward(mod, epsilon):
                    def patched_forward(hidden_states):
                        return triton_rmsnorm(hidden_states, mod.weight, eps=epsilon)
                    return patched_forward
                module.forward = make_patched_forward(module, eps)
                patched_count += 1
            else:
                print(f"WARNING: {name} has no weight, skipping")

    return patched_count


def inject_optimized_kernels(model) -> dict:
    """Inject custom Triton kernels into a transformers model."""
    stats = {'rmsnorm_modules': 0}
    stats['rmsnorm_modules'] = patch_rmsnorm_modules(model)
    return stats


# =============================================================================
# Main
# =============================================================================

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("Transformers Triton Kernel Injection (ROCm)")
    print("=" * 60)

    # Verify ROCm
    print(f"\nROCm HIP version: {getattr(torch.version, 'hip', 'N/A')}")
    print(f"GPU: {torch.cuda.get_device_name()}")

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"\n1. Loading model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    rmsnorm_count = sum(1 for _, m in model.named_modules() if 'RMSNorm' in type(m).__name__)
    print(f"   Found {rmsnorm_count} RMSNorm modules")

    print("\n2. Injecting optimized Triton kernels...")
    stats = inject_optimized_kernels(model)
    print(f"   RMSNorm modules patched: {stats['rmsnorm_modules']}")

    print("\n3. Verifying injection...")
    x = torch.randn(1, 10, model.config.hidden_size, device='cuda', dtype=torch.bfloat16)
    for name, module in model.named_modules():
        if 'RMSNorm' in type(module).__name__:
            out = module(x)
            print(f"   RMSNorm forward pass: {x.shape} -> {out.shape}")
            break

    print("\n4. Running generation test...")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    num_tokens = 50
    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    tokens_per_second = num_tokens / elapsed

    print(f"   Prompt: {prompt}")
    print(f"   Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    print(f"   Generated {num_tokens} tokens in {elapsed:.2f}s ({tokens_per_second:.1f} tokens/s)")

    print("\n" + "=" * 60)
    print("Success! Custom Triton kernels are being used on ROCm.")
    print("=" * 60)


if __name__ == "__main__":
    main()
