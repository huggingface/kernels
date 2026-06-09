#!/usr/bin/env python3
"""
Analyze a PyTorch operation to guide CPU kernel development.

Extracts compute/memory characteristics, identifies kernel type, and
recommends SIMD strategy and optimization approach.

Usage:
    python scripts/analyze_op.py --op "rms_norm" --shapes "1024x4096,2048x8192"
    python scripts/analyze_op.py --file baseline.py
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List


class OpAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze PyTorch model operations."""

    def __init__(self):
        self.operations = []
        self.shapes = {}
        self.dtypes = set()
        self.has_matmul = False
        self.has_linear = False
        self.activations = []
        self.reductions = []
        self.elementwise = []

    def visit_Call(self, node):
        """Visit function calls to identify operations."""
        if isinstance(node.func, ast.Attribute):
            if hasattr(node.func.value, "id") and node.func.value.id == "torch":
                op_name = node.func.attr
                self.operations.append(op_name)

                if op_name in ("matmul", "mm", "bmm"):
                    self.has_matmul = True
                elif op_name in ("sum", "mean", "max", "min", "norm"):
                    self.reductions.append(op_name)
                elif op_name in ("sigmoid", "tanh", "relu", "gelu", "silu"):
                    self.activations.append(op_name)
                elif op_name in ("clamp", "abs", "exp", "rsqrt", "sqrt"):
                    self.elementwise.append(op_name)

            elif hasattr(node.func.value, "attr"):
                if node.func.value.attr == "functional":
                    op_name = node.func.attr
                    self.operations.append(f"F.{op_name}")
                    if op_name in ("gelu", "relu", "silu", "softmax", "sigmoid"):
                        self.activations.append(op_name)
                    if op_name in ("linear",):
                        self.has_linear = True

        self.generic_visit(node)

    def visit_BinOp(self, node):
        """Visit binary operations."""
        op_map = {
            ast.Mult: "multiply",
            ast.Div: "divide",
            ast.Add: "add",
            ast.Sub: "subtract",
        }
        op_type = type(node.op)
        if op_type in op_map:
            self.elementwise.append(op_map[op_type])
        self.generic_visit(node)

    def visit_Assign(self, node):
        """Visit assignments to track nn.Linear."""
        if isinstance(node.value, ast.Call):
            if hasattr(node.value.func, "attr") and node.value.func.attr == "Linear":
                self.has_linear = True
        self.generic_visit(node)


def analyze_from_file(filepath: Path) -> Dict:
    """Analyze PyTorch file and extract optimization hints."""
    with open(filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    analyzer = OpAnalyzer()
    analyzer.visit(tree)

    # Extract shape information from module-level variables
    shapes = {}
    for line in source.split("\n"):
        if "=" in line and any(
            dim in line
            for dim in [
                "batch_size", "in_features", "out_features",
                "hidden_size", "input_size", "seq_len", "num_heads",
            ]
        ):
            match = re.match(r"(\w+)\s*=\s*(\d+)", line.strip())
            if match:
                shapes[match.group(1)] = int(match.group(2))

    return _build_analysis(analyzer, shapes)


def analyze_from_args(op_name: str, shapes_str: str) -> Dict:
    """Analyze operation from command-line args."""
    analyzer = OpAnalyzer()

    # Classify op name
    op_lower = op_name.lower().replace("_", "").replace("-", "")
    if op_lower in ("rmsnorm", "layernorm", "rms_norm", "layer_norm"):
        analyzer.reductions.append("norm")
        analyzer.elementwise.extend(["multiply", "rsqrt"])
    elif op_lower in ("softmax",):
        analyzer.reductions.extend(["max", "sum"])
        analyzer.elementwise.extend(["exp", "divide"])
    elif "gemm" in op_lower or "linear" in op_lower or "matmul" in op_lower:
        analyzer.has_matmul = True
    elif "attention" in op_lower or "flashatt" in op_lower or "flash_att" in op_lower:
        analyzer.has_matmul = True
        analyzer.reductions.append("softmax")
    elif "gelu" in op_lower or "silu" in op_lower or "relu" in op_lower:
        analyzer.activations.append(op_lower)
    elif "moe" in op_lower or "megablocks" in op_lower:
        analyzer.has_matmul = True

    # Parse shapes
    shapes = {}
    if shapes_str:
        for i, s in enumerate(shapes_str.split(",")):
            dims = s.strip().split("x")
            if len(dims) == 2:
                shapes[f"shape_{i}"] = f"{dims[0]}x{dims[1]}"
            elif len(dims) == 3:
                shapes[f"shape_{i}"] = f"{dims[0]}x{dims[1]}x{dims[2]}"

    return _build_analysis(analyzer, shapes)


def _build_analysis(analyzer: OpAnalyzer, shapes: Dict) -> Dict:
    """Build analysis result from analyzer state."""
    kernel_type = "unknown"
    if analyzer.has_matmul or analyzer.has_linear:
        if analyzer.activations or analyzer.elementwise:
            kernel_type = "gemm_fused"
        elif analyzer.reductions:
            kernel_type = "attention"
        else:
            kernel_type = "gemm"
    elif analyzer.reductions:
        kernel_type = "reduction"
    elif analyzer.elementwise or analyzer.activations:
        kernel_type = "elementwise"

    return {
        "kernel_type": kernel_type,
        "operations": analyzer.operations,
        "activations": analyzer.activations,
        "reductions": analyzer.reductions,
        "elementwise": analyzer.elementwise,
        "shapes": shapes,
        "has_gemm": analyzer.has_matmul or analyzer.has_linear,
    }


def print_analysis(analysis: Dict):
    """Pretty print the analysis results with CPU-specific recommendations."""

    print(f"\n{'=' * 70}")
    print(f"CPU Kernel Analysis")
    print(f"{'=' * 70}\n")

    print(f"Kernel Type: {analysis['kernel_type'].upper()}")
    print()

    if analysis["shapes"]:
        print("Shapes:")
        for key, val in analysis["shapes"].items():
            print(f"  {key}: {val}")
        print()

    print("Operations:")
    if analysis["has_gemm"]:
        print(f"  * GEMM/Linear detected")
    if analysis["activations"]:
        print(f"  * Activations: {', '.join(set(analysis['activations']))}")
    if analysis["reductions"]:
        print(f"  * Reductions: {', '.join(set(analysis['reductions']))}")
    if analysis["elementwise"]:
        print(f"  * Elementwise: {', '.join(set(analysis['elementwise']))}")
    print()

    # CPU-specific recommendations
    print("CPU Optimization Strategy:")

    if analysis["has_gemm"]:
        print("  Kernel Category: GEMM")
        print("  Architecture:")
        print("    - tinygemm path (M <= 4): fused dequant + _mm512_dpbf16_ps")
        print("    - brgemm path (M > 4): pre-dequant B + at::native::cpublas::brgemm()")
        print("  Threading: parallel_2d(m, n, compute_fn) with 2D factorization")
        print("  Compiler flags: -mavx512f -mavx512bf16 -mavx512vl -mavx512dq -mavx512bw")
        print("                  -mavx512vbmi -mfma -mf16c -fopenmp")
        print("  Note: AMX flags NOT needed — brgemm dispatches to AMX via oneDNN internally")
        print("  Key patterns: Unroll<N>, tinygemm_kernel_nn, cpu_types_avx512.hpp")
        print()
        print("  Reference files:")
        print("    - references/brgemm_patterns.yaml")
        print("    - references/quantized_gemm_patterns.yaml")
        print("    - references/threading_patterns.yaml")

    elif analysis["kernel_type"] in ("reduction", "elementwise"):
        print("  Kernel Category: Element-wise / Reduction")
        print("  Architecture: Direct AVX512 intrinsics (no brgemm)")
        print("  Threading: #pragma omp parallel for over rows")
        print("  Vectorization: FP32Vec16, BF16Vec32 abstractions")
        print("  Compiler flags: -mavx512f -mavx512bf16 -mavx512vl -mavx512dq -mavx512bw")
        print("                  -mavx512vbmi -mfma -mf16c -fopenmp")
        print("  Prefetch: _MM_HINT_T1 (L2)")
        print()
        print("  Reference files:")
        print("    - references/simd_optimization_patterns.yaml")
        print("    - references/memory_patterns.yaml")
        print("    - references/threading_patterns.yaml")

    elif analysis["kernel_type"] == "attention":
        print("  Kernel Category: Attention (Flash-Attention style)")
        print("  Architecture: Tiled attention with brgemm for Q@K and S@V")
        print("  Blocking: BLOCK_M=256, BLOCK_N=768 (attention-specific)")
        print("  Threading: parallel over batch * heads * M-tiles")
        print("  Requirements: AVX512 + AMX (via brgemm)")
        print()
        print("  Reference files:")
        print("    - references/brgemm_patterns.yaml")
        print("    - references/memory_patterns.yaml")

    print()
    print("File Structure:")
    print("  my_kernel_cpu/")
    print("  ├── cpu_features.hpp          # CPUID detection (own namespace)")
    print("  ├── my_kernel_cpu.cpp          # Dispatcher (runtime feature check)")
    print("  ├── my_kernel_cpu.hpp          # Shared declarations")
    print("  ├── my_kernel_cpu_torch.cpp    # Python ↔ C++ bridge")
    if not analysis["has_gemm"]:
        print("  ├── my_kernel_avx2.cpp         # AVX2 implementation (optional)")
        print("  ├── my_kernel_avx2.hpp")
    print("  ├── my_kernel_avx512.cpp       # AVX512 implementation")
    print("  └── my_kernel_avx512.hpp")
    print()
    print("  torch_binding.cpp              # Op registration (registration.h)")
    print("  build.toml                     # Multi-target compilation config")
    print()

    print("Relevant Reference Files:")
    print("  - references/runtime_dispatch.yaml       # cpu_features.hpp pattern")
    print("  - references/correctness.yaml             # Critical constraints")
    print("  - references/implementation_reference.md  # C++ templates")
    if analysis["has_gemm"]:
        print("  - references/brgemm_patterns.yaml         # brgemm API")
        print("  - references/quantized_gemm_patterns.yaml # 4-bit GEMM")
    print("  - references/optimization_levels.yaml     # Progressive optimization")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze PyTorch op for CPU kernel development")
    parser.add_argument("--op", type=str, help="Operation name (e.g., rms_norm, flash_attention)")
    parser.add_argument("--shapes", type=str, default="", help="Shapes as MxN,MxN (e.g., 1024x4096,2048x8192)")
    parser.add_argument("--file", type=Path, help="PyTorch baseline file to analyze")
    args = parser.parse_args()

    if args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        analysis = analyze_from_file(args.file)
    elif args.op:
        analysis = analyze_from_args(args.op, args.shapes)
    else:
        print("Error: Provide either --op or --file")
        sys.exit(1)

    print_analysis(analysis)


if __name__ == "__main__":
    main()
