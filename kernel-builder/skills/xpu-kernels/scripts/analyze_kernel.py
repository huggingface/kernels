#!/usr/bin/env python3
"""
Analyze PyTorch kernel to extract structure and guide Triton optimization.

Usage:
    python scripts/analyze_kernel.py <pytorch_file>
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class KernelAnalyzer(ast.NodeVisitor):
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
        # torch.matmul
        if isinstance(node.func, ast.Attribute):
            if hasattr(node.func.value, "id") and node.func.value.id == "torch":
                op_name = node.func.attr
                self.operations.append(op_name)

                if op_name == "matmul":
                    self.has_matmul = True
                elif op_name in ["sum", "mean", "max", "min"]:
                    self.reductions.append(op_name)
                elif op_name in ["sigmoid", "tanh", "relu", "gelu", "silu"]:
                    self.activations.append(op_name)
                elif op_name == "clamp":
                    self.elementwise.append("clamp")

            # torch.nn.functional
            elif hasattr(node.func.value, "attr"):
                if node.func.value.attr == "functional":
                    op_name = node.func.attr
                    self.operations.append(f"F.{op_name}")
                    if op_name in ["gelu", "relu", "silu", "softmax", "sigmoid"]:
                        self.activations.append(op_name)

        self.generic_visit(node)

    def visit_BinOp(self, node):
        """Visit binary operations (*, /, +, -)."""
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


def analyze_pytorch_kernel(filepath: Path) -> Dict:
    """Analyze PyTorch kernel file and extract optimization hints."""

    with open(filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    analyzer = KernelAnalyzer()
    analyzer.visit(tree)

    # Extract shape information from module-level variables
    shapes = {}
    for line in source.split("\n"):
        if "=" in line and any(
            dim in line
            for dim in ["batch_size", "in_features", "out_features", "hidden_size", "input_size"]
        ):
            match = re.match(r"(\w+)\s*=\s*(\d+)", line.strip())
            if match:
                shapes[match.group(1)] = int(match.group(2))

    # Determine kernel type
    kernel_type = "unknown"
    if analyzer.has_matmul or analyzer.has_linear:
        if analyzer.activations or analyzer.elementwise:
            kernel_type = "gemm_epilogue"
        elif analyzer.reductions:
            kernel_type = "gemm_reduction"
        else:
            kernel_type = "gemm"
    elif analyzer.reductions:
        kernel_type = "reduction"
    elif analyzer.elementwise:
        kernel_type = "elementwise"

    # Fusion analysis
    fusion_opportunities = []
    if analyzer.has_matmul or analyzer.has_linear:
        if len(analyzer.activations) <= 2 and len(analyzer.elementwise) <= 3:
            fusion_opportunities.append("Light epilogue fusion (GEMM + activation + elementwise)")
        else:
            fusion_opportunities.append("Heavy epilogue - consider partial fusion or split")

        if analyzer.reductions:
            fusion_opportunities.append(
                "WARNING: GEMM + reduction - use 2D GEMM then separate reduction kernel"
            )

    # Memory pattern recommendation
    memory_pattern = "block_pointers"  # default
    if "Stream K" in str(analyzer.operations) or len(analyzer.reductions) > 1:
        memory_pattern = "tensor_descriptors"

    return {
        "kernel_type": kernel_type,
        "operations": analyzer.operations,
        "activations": analyzer.activations,
        "reductions": analyzer.reductions,
        "elementwise": analyzer.elementwise,
        "shapes": shapes,
        "fusion_opportunities": fusion_opportunities,
        "memory_pattern": memory_pattern,
        "has_gemm": analyzer.has_matmul or analyzer.has_linear,
    }


def print_analysis(analysis: Dict, filepath: Path):
    """Pretty print the analysis results."""

    print(f"\n{'=' * 70}")
    print(f"Analysis: {filepath.name}")
    print(f"{'=' * 70}\n")

    print(f"Kernel Type: {analysis['kernel_type'].upper()}")
    print(f"Memory Pattern: {analysis['memory_pattern']}")
    print()

    if analysis["shapes"]:
        print("Shapes:")
        for key, val in analysis["shapes"].items():
            print(f"  {key}: {val}")
        print()

    print("Operations:")
    print(f"  Total: {len(analysis['operations'])}")
    if analysis["has_gemm"]:
        print(f"  ✓ GEMM/Linear")
    if analysis["activations"]:
        print(f"  ✓ Activations: {', '.join(set(analysis['activations']))}")
    if analysis["reductions"]:
        print(f"  ✓ Reductions: {', '.join(set(analysis['reductions']))}")
    if analysis["elementwise"]:
        print(f"  ✓ Elementwise: {', '.join(set(analysis['elementwise']))}")
    print()

    if analysis["fusion_opportunities"]:
        print("Fusion Opportunities:")
        for opp in analysis["fusion_opportunities"]:
            if "WARNING" in opp:
                print(f"  ⚠️  {opp}")
            else:
                print(f"  → {opp}")
        print()

    # Recommendations
    print("Recommended Optimizations:")

    if analysis["has_gemm"]:
        print("  1. Use tensor descriptors (preferred on XPU) or block pointers")
        print("  2. Apply tile swizzling (GROUP_SIZE_M)")

        # Tile size recommendations based on shape
        batch_size = analysis["shapes"].get("batch_size", 0)
        if batch_size and batch_size < 256:
            print("  3. Use smaller BLOCK_M (32-64) for skinny M")
        else:
            print("  3. Try large tiles (256x256) with autotune")

        print("  4. Autotune: num_warps={4,8,16,32}, grf_mode='256'")
        print("  5. Mixed precision: bf16/fp16 inputs, fp32 accumulator")
        print("  6. Pre-pack weight transpose: weight_t = weight.t().contiguous()")

    if "sigmoid" in analysis["activations"]:
        print("  → Use exp2-based sigmoid (faster on XPU)")

    if "tanh" in analysis["activations"]:
        print("  → Implement tanh via sigmoid: tanh(x) = 2*sigmoid(2x) - 1")

    if "gelu" in analysis["activations"]:
        print("  → Use tanh-approximation GeLU with JIT helper")

    if analysis["reductions"]:
        if analysis["has_gemm"]:
            print("  ⚠️  Split GEMM and reduction into separate kernels")
            print("     (Don't serialize over N tiles inside one program)")
        else:
            print("  → Use multi-row tiling for reductions")
            print("  → Query max_work_group_size for BLOCK_SIZE_Y")

    print()

    print("Relevant Reference Files:")
    print("  • references/xpu_optimizations.yaml - Core XPU patterns")
    if analysis["fusion_opportunities"]:
        print("  • references/fusion_patterns.yaml - Fusion guidelines")
    print("  • references/memory_patterns.yaml - Memory access patterns")
    print("  • references/correctness.yaml - Critical constraints")
    print()

    # Template suggestion
    if analysis["kernel_type"] == "gemm":
        print("Suggested Template: See GEMM pattern in references/implementation_reference.md")
    elif analysis["kernel_type"] == "gemm_epilogue":
        print("Suggested Template: See GEMM with epilogue pattern in references/implementation_reference.md")
    elif analysis["kernel_type"] in ["reduction", "gemm_reduction"]:
        print("Suggested Template: See reduction pattern in references/implementation_reference.md")
    print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_kernel.py <pytorch_file>")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    analysis = analyze_pytorch_kernel(filepath)
    print_analysis(analysis, filepath)


if __name__ == "__main__":
    main()
