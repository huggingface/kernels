#!/usr/bin/env python3
"""
Minimal example: Inject custom CUDA kernels into HuggingFace Transformers models.

This script demonstrates the essential pattern for integrating custom CUDA kernels
with transformers models like LLaMA, Mistral, and Qwen.

Key lessons:
1. Transformers RMSNorm modules always have weights (unlike some diffusers modules)
2. Use 'RMSNorm' substring match to catch LlamaRMSNorm, MistralRMSNorm, etc.
3. Check for 'variance_epsilon' (LLaMA) or 'eps' (others) for epsilon value
4. Use Flash Attention 2 for attention optimization instead of custom processors

Usage:
    cd examples/ltx_video
    uv pip install -e .  # Build kernels first
    python ../../.claude/skills/h100-diffusers-kernels/scripts/transformers_injection_example.py
"""

import sys
import time

import torch
import torch.nn as nn

# Add kernel path (adjust based on your location)
sys.path.insert(0, "torch-ext")

from ltx_kernels import rmsnorm


# =============================================================================
# Step 1: RMSNorm Module Patcher
# =============================================================================

def patch_rmsnorm_modules(model: nn.Module) -> int:
    """
    Patch all RMSNorm modules to use custom CUDA kernel.

    Works with LlamaRMSNorm, MistralRMSNorm, Qwen2RMSNorm, etc.

    IMPORTANT: Unlike diffusers, transformers RMSNorm always has weights.
    """
    patched_count = 0

    for name, module in model.named_modules():
        class_name = type(module).__name__

        # Match all RMSNorm variants (LlamaRMSNorm, MistralRMSNorm, etc.)
        if 'RMSNorm' in class_name:
            # Get epsilon - LLaMA uses 'variance_epsilon', others use 'eps'
            eps = getattr(module, 'variance_epsilon', None)
            if eps is None:
                eps = getattr(module, 'eps', 1e-6)

            # Transformers RMSNorm always has weight
            has_weight = hasattr(module, 'weight') and module.weight is not None

            if has_weight:
                def make_patched_forward(mod, epsilon):
                    def patched_forward(hidden_states):
                        return rmsnorm(hidden_states, mod.weight, eps=epsilon)
                    return patched_forward
                module.forward = make_patched_forward(module, eps)
                patched_count += 1
            else:
                print(f"WARNING: {name} has no weight, skipping")

    return patched_count


# =============================================================================
# Step 2: Kernel Injection Function
# =============================================================================

def inject_optimized_kernels(model) -> dict:
    """
    Inject custom CUDA kernels into a transformers model.

    Call this AFTER loading model to CUDA.
    """
    stats = {'rmsnorm_modules': 0}

    # Patch RMSNorm modules
    stats['rmsnorm_modules'] = patch_rmsnorm_modules(model)

    return stats


# =============================================================================
# Step 3: Main - Demonstrate the Pattern
# =============================================================================

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("Transformers Kernel Injection Example")
    print("=" * 60)

    # Model to use - change as needed
    # Options: "meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", etc.
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for demo

    # Load model
    print(f"\n1. Loading model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Count RMSNorm modules before patching
    rmsnorm_count = sum(1 for _, m in model.named_modules() if 'RMSNorm' in type(m).__name__)
    print(f"   Found {rmsnorm_count} RMSNorm modules")

    # Inject kernels
    print("\n2. Injecting optimized CUDA kernels...")
    stats = inject_optimized_kernels(model)
    print(f"   RMSNorm modules patched: {stats['rmsnorm_modules']}")

    # Verify injection worked
    print("\n3. Verifying injection...")
    x = torch.randn(1, 10, model.config.hidden_size, device='cuda', dtype=torch.bfloat16)
    for name, module in model.named_modules():
        if 'RMSNorm' in type(module).__name__:
            out = module(x)
            print(f"   RMSNorm forward pass: {x.shape} -> {out.shape}")
            break

    # Run generation test
    print("\n4. Running generation test...")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Warmup
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Benchmark
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
    print("Success! Custom kernels are being used.")
    print("=" * 60)


if __name__ == "__main__":
    main()
