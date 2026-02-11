#!/usr/bin/env python3
"""
Minimal example: Inject custom CUDA kernels into LTX-Video pipeline.

This script demonstrates the essential pattern for integrating custom CUDA kernels
with diffusers pipelines. For full usage, see examples/ltx_video/generate_video.py.

Key lessons:
1. Check if RMSNorm has weight (elementwise_affine may be False)
2. Use type(module).__name__ not isinstance() to detect diffusers modules
3. LTX-Video uses GELU, not GEGLU - check your target model
4. Inject kernels AFTER loading to CUDA, BEFORE CPU offloading

Usage:
    cd examples/ltx_video
    uv pip install -e .  # Build kernels first
    python ../../.claude/skills/h100-diffusers-kernels/references/ltx_kernel_injection_example.py
"""

import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Add kernel path (adjust based on your location)
sys.path.insert(0, "torch-ext")

from ltx_kernels import rmsnorm


# =============================================================================
# Step 1: Custom Attention Processor
# =============================================================================

class OptimizedLTXVideoAttnProcessor:
    """
    Attention processor using custom rmsnorm kernel for Q/K normalization.

    NOTE: attn.norm_q and attn.norm_k HAVE weights (elementwise_affine=True).
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Import here to avoid issues if diffusers not installed
        from diffusers.models.transformers.transformer_ltx import apply_rotary_emb
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Q, K, V projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # >>> CUSTOM KERNEL: RMSNorm for Q/K normalization <<<
        query = rmsnorm(query, attn.norm_q.weight, eps=attn.norm_q.eps)
        key = rmsnorm(key, attn.norm_k.weight, eps=attn.norm_k.eps)

        # Apply rotary embeddings (computed by diffusers)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Reshape for multi-head attention
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Dispatch attention
        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# =============================================================================
# Step 2: RMSNorm Module Patcher
# =============================================================================

def patch_rmsnorm_modules(model: nn.Module) -> int:
    """
    Patch all RMSNorm modules to use custom CUDA kernel.

    IMPORTANT: Handles modules with AND without weights!
    - LTX transformer blocks: elementwise_affine=False (NO weight)
    - LTX attention norm_q/k: elementwise_affine=True (HAS weight)
    """
    patched_count = 0

    for name, module in model.named_modules():
        # Use class name check (not isinstance) to catch diffusers RMSNorm
        if type(module).__name__ == 'RMSNorm':
            eps = getattr(module, 'eps', 1e-6)
            has_weight = hasattr(module, 'weight') and module.weight is not None

            if has_weight:
                # Module HAS learnable weight
                def make_patched_forward_with_weight(mod, epsilon):
                    def patched_forward(x):
                        return rmsnorm(x, mod.weight, eps=epsilon)
                    return patched_forward
                module.forward = make_patched_forward_with_weight(module, eps)
            else:
                # Module has NO weight (elementwise_affine=False)
                def make_patched_forward_no_weight(epsilon):
                    def patched_forward(x):
                        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
                        return rmsnorm(x, weight, eps=epsilon)
                    return patched_forward
                module.forward = make_patched_forward_no_weight(eps)

            patched_count += 1

    return patched_count


# =============================================================================
# Step 3: Kernel Injection Function
# =============================================================================

def inject_optimized_kernels(pipe) -> dict:
    """
    Inject custom CUDA kernels into the LTX-Video pipeline.

    Call this AFTER pipe.to("cuda"), BEFORE pipe.enable_model_cpu_offload().
    """
    stats = {'attention_processors': 0, 'rmsnorm_modules': 0}

    if not hasattr(pipe, 'transformer'):
        print("WARNING: Pipeline has no 'transformer' attribute!")
        return stats

    transformer = pipe.transformer

    # Replace attention processors with optimized version
    for name, module in transformer.named_modules():
        if hasattr(module, 'set_processor') and hasattr(module, 'processor'):
            module.set_processor(OptimizedLTXVideoAttnProcessor())
            stats['attention_processors'] += 1

    # Patch RMSNorm modules
    stats['rmsnorm_modules'] = patch_rmsnorm_modules(transformer)

    return stats


# =============================================================================
# Step 4: Main - Demonstrate the Pattern
# =============================================================================

def main():
    from diffusers import LTXPipeline
    from diffusers.utils import export_to_video

    print("=" * 60)
    print("LTX-Video Kernel Injection Example")
    print("=" * 60)

    # Load pipeline
    print("\n1. Loading pipeline...")
    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    # Inject kernels (BEFORE CPU offloading!)
    print("\n2. Injecting optimized CUDA kernels...")
    stats = inject_optimized_kernels(pipe)
    print(f"   Attention processors replaced: {stats['attention_processors']}")
    print(f"   RMSNorm modules patched: {stats['rmsnorm_modules']}")

    # Verify injection worked
    print("\n3. Verifying injection...")
    for name, module in pipe.transformer.named_modules():
        if hasattr(module, 'processor'):
            processor_name = type(module.processor).__name__
            assert processor_name == 'OptimizedLTXVideoAttnProcessor', \
                f"Expected OptimizedLTXVideoAttnProcessor, got {processor_name}"
            print(f"   ✓ Attention processor: {processor_name}")
            break

    # Test RMSNorm forward pass
    x = torch.randn(1, 10, 2048, device='cuda', dtype=torch.bfloat16)
    for name, module in pipe.transformer.named_modules():
        if type(module).__name__ == 'RMSNorm':
            out = module(x)
            print(f"   ✓ RMSNorm forward: {x.shape} -> {out.shape}")
            break

    # Enable memory optimization (AFTER injection)
    print("\n4. Enabling CPU offloading...")
    pipe.enable_model_cpu_offload()

    # Generate a short test video
    print("\n5. Generating test video (9 frames, 5 steps)...")
    output = pipe(
        prompt="A cat sleeping in warm sunlight",
        num_frames=9,
        height=480,
        width=704,
        num_inference_steps=5,  # Low for quick test
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    # Save
    output_path = "test_output.mp4"
    export_to_video(output.frames[0], output_path, fps=8)
    print(f"\n✓ Video saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Success! Custom kernels are being used.")
    print("=" * 60)


if __name__ == "__main__":
    main()
