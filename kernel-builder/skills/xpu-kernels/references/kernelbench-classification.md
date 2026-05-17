# KernelBench Operator Classification & Skill Mapping

This document classifies KernelBench operators into categories and maps each to the appropriate kernel skill/pattern.

## Classification Taxonomy

### Level 1: Basic Operators (53 operators)

#### Category A: GEMM / Matrix Multiplication (18 operators)

| ID | Name | Sub-type | Key Skill |
|----|------|----------|-----------|
| 1 | Square matrix multiplication | Dense GEMM | Tile Swizzle + Autotune |
| 2 | Standard matrix multiplication | Dense GEMM (M!=N) | Tile Swizzle + Autotune |
| 3 | Batched matrix multiplication | BMM | Batch-indexed GEMM |
| 4 | Matrix-vector multiplication | MatVec | 1D reduction pattern |
| 5 | Matrix-scalar multiplication | Elementwise | Scale kernel |
| 6 | Matmul with large K | Large-K GEMM | K-dimension blocking |
| 7 | Matmul with small K | Small-K GEMM | Fewer K-iterations |
| 8 | Matmul with irregular shapes | Non-square GEMM | Mask handling |
| 9 | Tall-skinny matmul | Tall-skinny GEMM | Tile shape tuning |
| 10 | 3D tensor-matrix mul | Batched GEMM | Reshape + GEMM |
| 11 | 4D tensor-matrix mul | Batched GEMM | Einsum decomposition |
| 12 | Diagonal matrix mul | Special GEMM | Elementwise pattern |
| 13 | Symmetric matrices | Dense GEMM | Standard GEMM |
| 14 | Upper triangular mul | Masked GEMM | Triangle mask |
| 15 | Lower triangular mul | Masked GEMM | Triangle mask |
| 16 | Transposed A | Transposed GEMM | Stride adjustment |
| 17 | Transposed B | Transposed GEMM | Stride adjustment |
| 18 | Both transposed | Transposed GEMM | Stride adjustment |

**Key Pattern**: Template 5 (GEMM with Tile Swizzle)
**Critical Optimization**: Tile swizzle + L2 cache grouping + tensor descriptors

#### Category B: Elementwise / Activation Functions (14 operators)

| ID | Name | Sub-type | Key Skill |
|----|------|----------|-----------|
| 19 | ReLU | Branching | `tl.where(x > 0, x, 0)` |
| 20 | LeakyReLU | Branching | `tl.where(x > 0, x, alpha*x)` |
| 21 | Sigmoid | Transcendental | `1/(1+exp(-x))` |
| 22 | Tanh | Transcendental | `(exp(2x)-1)/(exp(2x)+1)` |
| 23 | Softmax | Row reduction | Online softmax |
| 24 | LogSoftmax | Row reduction | Online softmax + log |
| 25 | Swish/SiLU | Transcendental | `x * sigmoid(x)` |
| 26 | GELU | Transcendental | `0.5*x*(1+erf(x/sqrt(2)))` |
| 27 | SELU | Branching + exp | `scale * where(x>0, x, alpha*(exp(x)-1))` |
| 28 | HardSigmoid | Clamp | `clamp((x+3)/6, 0, 1)` |
| 29 | Softplus | Transcendental | `log(1+exp(x))` |
| 30 | Softsign | Division | `x/(1+abs(x))` |
| 31 | ELU | Branching + exp | `where(x>0, x, alpha*(exp(x)-1))` |
| 32 | HardTanh | Clamp | `clamp(x, -1, 1)` |

**Key Pattern**: Template 1 (Elementwise)
**Critical Optimization**: Large BLOCK_SIZE (4096-16384), FP32 compute

#### Category C: Normalization (8 operators)

| ID | Name | Sub-type | Key Skill |
|----|------|----------|-----------|
| 33 | BatchNorm | Multi-dim reduction | Welford algorithm |
| 34 | InstanceNorm | Per-instance reduction | Per-sample norm |
| 35 | GroupNorm | Group reduction | Grouped channels |
| 36 | RMSNorm | Row reduction | `x * rsqrt(mean(x^2) + eps)` |
| 37 | FrobeniusNorm | Full reduction | `sqrt(sum(x^2))` |
| 38 | L1 Norm | Full reduction | `sum(abs(x))` |
| 39 | L2 Norm | Full reduction | `sqrt(sum(x^2))` |
| 40 | LayerNorm | Row reduction | `(x-mean)/std * w + b` |

**Key Pattern**: Template 3 (Row-wise Reduction)
**Critical Optimization**: FP32 accumulation, proper reduction

#### Category D: Pooling (6 operators)

| ID | Name | Sub-type | Key Skill |
|----|------|----------|-----------|
| 41 | Max Pooling 1D | Sliding window | Max reduction |
| 42 | Max Pooling 2D | 2D window | 2D index mapping |
| 43 | Max Pooling 3D | 3D window | Program_id flattening |
| 44 | Average Pooling 1D | Sliding window | Sum + divide |
| 45 | Average Pooling 2D | 2D window | 2D index mapping |
| 46 | Average Pooling 3D | 3D window | Program_id flattening |

**Key Challenge**: 3D grid mapping with Triton's program_id limits

#### Category E: Reduction (7 operators)

| ID | Name | Sub-type | Key Skill |
|----|------|----------|-----------|
| 47 | Sum reduction | Sum | `tl.sum()` |
| 48 | Mean reduction | Mean | `tl.sum() / count` |
| 49 | Max reduction | Max | `tl.max()` |
| 50 | Min reduction | Min | `tl.min()` |
| 51 | Argmax | Index + max | Two-pass or manual |
| 52 | Argmin | Index + min | Two-pass or manual |
| 53 | Min (duplicate) | Min | `tl.min()` |

**Key Pattern**: Template 5 (Dimension Reduction)
**Key Challenge**: Argmax/Argmin require manual implementation

### Level 2: Fused Operators (20+ operators)

Combine multiple operations into single kernels.

| Category | Examples | Strategy |
|----------|---------|----------|
| GEMM + Activation | Gemm_ReLU, Gemm_GELU | Fuse activation into GEMM epilogue |
| GEMM + Norm | Gemm_BatchNorm, Gemm_GroupNorm | Two-phase kernel |
| GEMM + Scale | Gemm_Scale, Gemm_Divide | Fuse into GEMM store |
| Multi-op fusion | Matmul_Sum_Max_AvgPool | Sequential fusion |

**Key Pattern**: Template 6 (Fused GEMM + Activation)

### Level 3-4: Network Models / Transformers

Full models requiring multiple kernel types. Decompose into Level 1 operators.

### Level 6-7: Advanced / Expert

| Operator | Type | Strategy |
|----------|------|----------|
| MinGPTNewGelu | Fused activation | GELU approximation kernel |
| ScaledDotProductAttention | Attention | Flash Attention pattern |
| GELU_And_Mul | Fused activation | `gelu(x) * y` |
| MoE_TopK_Softmax | MoE routing | Specialized kernel |
| Gemm_A8W8_Blockwise | Quantized GEMM | INT8 with block scaling |

## Category → Skill Mapping

| Category | Skill File | Priority |
|----------|-----------|----------|
| **GEMM** | `gemm-skill.md` (planned) | P0 - Most impactful |
| **Elementwise** | `elementwise-skill.md` (planned) | P0 - Most common |
| **Normalization** | `normalization-skill.md` (planned) | P1 - Frequently used |
| **Reduction** | `reduction-skill.md` (planned) | P1 - Common pattern |
| **Softmax** | `softmax-skill.md` (planned) | P1 - Critical for attention |
| **Pooling** | `pooling-skill.md` (planned) | P2 - Moderate complexity |
| **Attention** | `attention-skill.md` (planned) | P2 - High complexity |
| **Fused** | `fused-skill.md` (planned) | P2 - Combination patterns |

## Performance Expectations by Category

Based on kernel-agent test results:

| Category | Achievable Speedup | Difficulty | Notes |
|----------|-------------------|------------|-------|
| Elementwise | 1.0-3.0x | Low | Large blocks, memory-bound |
| Reduction (sum/mean) | 1.5-5.0x | Medium | Good parallelism |
| Pooling | 1.5-5.0x | Medium | Grid mapping challenge |
| LayerNorm/RMSNorm | 1.5-2.0x | Medium | Row-wise reduction |
| Dense GEMM | 0.8-1.2x | High | Tile swizzle critical |
| Batched GEMM | 0.6-0.9x | High | Memory bandwidth limited |
| BatchNorm | <0.1x | Very High | HIP sync issues |
| Argmax/Argmin | FAIL | Very High | Triton API limitation |
| Fused operators | 0.3-1.0x | Very High | Correctness challenges |

## Recommended Skill Development Order

1. **Phase 1 (Quick wins)**: Elementwise activations, Sum/Mean reduction
2. **Phase 2 (Core)**: GEMM with tile swizzle, LayerNorm/RMSNorm
3. **Phase 3 (Advanced)**: Softmax, Pooling, Attention
4. **Phase 4 (Expert)**: Fused operators, BatchNorm, Quantized GEMM
