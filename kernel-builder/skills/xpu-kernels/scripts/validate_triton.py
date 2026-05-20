#!/usr/bin/env python3
"""
Validate Triton kernel for common XPU optimization issues.

Usage:
    python scripts/validate_triton.py <triton_file>
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


class ValidationError:
    def __init__(self, level: str, message: str, line_num: int = None):
        self.level = level  # 'ERROR', 'WARNING', 'INFO'
        self.message = message
        self.line_num = line_num

    def __str__(self):
        prefix = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}[self.level]
        loc = f" (line {self.line_num})" if self.line_num else ""
        return f"{prefix} {self.level}: {self.message}{loc}"


def validate_triton_kernel(filepath: Path) -> List[ValidationError]:
    """Validate Triton kernel against XPU optimization guidelines."""

    with open(filepath, "r") as f:
        source = f.read()

    lines = source.split("\n")
    errors = []

    # 1. Check for autotune parameter defaults (CRITICAL)
    autotune_params = set()
    in_autotune = False
    for i, line in enumerate(lines):
        if "@triton.autotune" in line:
            in_autotune = True
        if in_autotune and "Config" in line:
            # Extract parameter names from Config dict
            matches = re.findall(r"'(\w+)':", line)
            autotune_params.update(matches)
        if in_autotune and "@triton.jit" in line:
            in_autotune = False

    # Check kernel signature for defaults on autotune params
    in_kernel_sig = False
    for i, line in enumerate(lines):
        if "@triton.jit" in line:
            in_kernel_sig = True
        if in_kernel_sig and "def " in line and "(" in line:
            in_kernel_sig = True
        if in_kernel_sig:
            for param in autotune_params:
                if f"{param}:" in line and "=" in line:
                    errors.append(
                        ValidationError(
                            "ERROR",
                            f"Autotune parameter '{param}' has default value in kernel signature. "
                            f"This causes 'Conflicting meta-parameters' error. Remove the default.",
                            i + 1,
                        )
                    )
            if ")" in line and in_kernel_sig:
                break

    # 2. Check grid dimensionality with swizzling
    has_swizzling = "GROUP_SIZE_M" in source or "swizzle" in source.lower()
    has_2d_grid = False
    for i, line in enumerate(lines):
        if "grid" in line and "=" in line:
            # Check for 2D grid pattern: (triton.cdiv(...), triton.cdiv(...))
            if line.count("triton.cdiv") >= 2 and line.count(",") >= 1:
                # Try to detect if it's a tuple with 2+ elements
                if "(" in line and ")" in line:
                    tuple_content = line[line.index("(") : line.rindex(")") + 1]
                    # Count commas outside of nested parens
                    paren_depth = 0
                    comma_count = 0
                    for char in tuple_content:
                        if char == "(":
                            paren_depth += 1
                        elif char == ")":
                            paren_depth -= 1
                        elif char == "," and paren_depth == 1:
                            comma_count += 1
                    if comma_count >= 1:
                        has_2d_grid = True
                        if has_swizzling:
                            errors.append(
                                ValidationError(
                                    "ERROR",
                                    "Grid is 2D but tile swizzling is used. Grid must be 1D "
                                    "when using GROUP_SIZE_M swizzling.",
                                    i + 1,
                                )
                            )

    # 3. Check boundary_check format
    for i, line in enumerate(lines):
        if "boundary_check" in line:
            # Check if it's using booleans instead of dimension indices
            if "True" in line or "False" in line:
                errors.append(
                    ValidationError(
                        "ERROR",
                        "boundary_check uses booleans. Use dimension indices (0, 1) instead.",
                        i + 1,
                    )
                )
            # Check if it's a descriptor load (descriptors don't support boundary_check)
            if ".load(" in line and "boundary_check" in line:
                # Check if this is a descriptor (desc.load pattern)
                # Look backwards for descriptor creation
                for j in range(max(0, i - 20), i):
                    if "make_tensor_descriptor" in lines[j]:
                        errors.append(
                            ValidationError(
                                "ERROR",
                                "Tensor descriptor .load() does NOT accept boundary_check parameter. "
                                "Remove it - descriptors handle boundaries internally.",
                                i + 1,
                            )
                        )
                        break

    # 4. Check for float64 usage (CRITICAL performance issue)
    for i, line in enumerate(lines):
        if "float64" in line.lower() or "tl.float64" in line:
            errors.append(
                ValidationError(
                    "WARNING",
                    "float64 detected. This is 5-10x slower on XPU. Use float32 unless absolutely required.",
                    i + 1,
                )
            )

    # 5. Check for int32 overflow in batch offset calculations
    batch_offset_pattern = r"(program_id|pid|bid)\s*\*\s*stride"
    for i, line in enumerate(lines):
        if re.search(batch_offset_pattern, line):
            if ".to(tl.int64)" not in line and "to(tl.int64)" not in line:
                errors.append(
                    ValidationError(
                        "WARNING",
                        "Batch offset calculation may overflow int32. Cast program_id to int64: "
                        "offset = bid.to(tl.int64) * stride",
                        i + 1,
                    )
                )

    # 6. Check for num_warps=32 without autotune
    for i, line in enumerate(lines):
        if "num_warps=32" in line or "num_warps = 32" in line:
            # Check if it's in a single Config (not autotuned)
            if "@triton.autotune" not in source or source.count("num_warps=32") == 1:
                errors.append(
                    ValidationError(
                        "WARNING",
                        "num_warps=32 used without autotuning. This can hurt performance on "
                        "skinny-M or heavy-epilogue kernels. Sweep {4,8,16,32}.",
                        i + 1,
                    )
                )

    # 7. Check for mixed block pointer and tensor descriptor APIs
    has_block_ptr = "make_block_ptr" in source
    has_tensor_desc = "make_tensor_descriptor" in source
    if has_block_ptr and has_tensor_desc:
        errors.append(
            ValidationError(
                "INFO",
                "Both block pointers and tensor descriptors found. This is OK if used for "
                "different operations (e.g., descriptors for loads, manual pointers for atomics). "
                "But do NOT mix APIs for the same load/store operation.",
            )
        )

    # 8. Check for .item() or device-to-host sync in hot path
    for i, line in enumerate(lines):
        if ".item()" in line or "float(tensor" in line or "int(tensor" in line:
            if "def forward" in "".join(lines[max(0, i - 20) : i]):
                errors.append(
                    ValidationError(
                        "ERROR",
                        "Device-to-host sync (.item() or float(tensor)) detected in forward pass. "
                        "This forces synchronization and kills performance.",
                        i + 1,
                    )
                )

    # 9. Check for weight transpose in forward() hot path
    for i, line in enumerate(lines):
        if ".t()" in line and ".contiguous()" in line:
            # Check if inside forward()
            in_forward = False
            for j in range(max(0, i - 20), i):
                if "def forward" in lines[j]:
                    in_forward = True
                    break
            if in_forward:
                errors.append(
                    ValidationError(
                        "WARNING",
                        "Weight transpose (.t().contiguous()) in forward() hot path. "
                        "Pre-pack once and cache to avoid per-iteration overhead.",
                        i + 1,
                    )
                )

    # 10. Check for GEMM with reduction loop over N (serialization issue)
    has_n_loop = False
    for i, line in enumerate(lines):
        if re.search(r"for.*range\(.*,\s*N\s*[,)]", line):
            has_n_loop = True
            if (
                "tl.dot"
                in source[max(0, source.rfind("\n", 0, i) - 500) : source.find("\n", i) + 500]
            ):
                errors.append(
                    ValidationError(
                        "ERROR",
                        "GEMM kernel loops over N tiles inside one program. This serializes "
                        "parallelism. Use 2D grid (pid_m, pid_n) instead.",
                        i + 1,
                    )
                )

    # 11. Check for tl.exp usage (prefer tl.math.exp2 on XPU)
    for i, line in enumerate(lines):
        if "tl.exp(" in line and "tl.math.exp2" not in source:
            errors.append(
                ValidationError(
                    "INFO",
                    "tl.exp() found. Consider using exp2-based implementation for better XPU performance: "
                    "exp(x) = exp2(x * 1.44269504)",
                    i + 1,
                )
            )

    # 12. Check for get_inputs / get_init_inputs (benchmark harness interface)
    has_model_class = "class Model" in source
    if has_model_class:
        if "def get_inputs" not in source:
            errors.append(
                ValidationError(
                    "WARNING",
                    "Model class found but no get_inputs() function. This is required by the "
                    "benchmark harness (ai-bench).",
                )
            )
        if "def get_init_inputs" not in source:
            errors.append(
                ValidationError(
                    "WARNING",
                    "Model class found but no get_init_inputs() function. This is required by the "
                    "benchmark harness (ai-bench).",
                )
            )

    # 15. Success indicators
    if not errors:
        errors.append(ValidationError("INFO", "No critical issues found! ✓"))

    # Positive feedback for good patterns
    if has_block_ptr or has_tensor_desc:
        errors.append(
            ValidationError(
                "INFO", "✓ Using modern memory access API (block pointers or tensor descriptors)"
            )
        )
    if "bfloat16" in source or "float16" in source:
        errors.append(ValidationError("INFO", "✓ Using reduced precision inputs (bf16/fp16)"))
    if "float32" in source and "accumulator" in source.lower():
        errors.append(ValidationError("INFO", "✓ Using fp32 accumulator for numerical stability"))
    return errors


def print_validation_results(errors: List[ValidationError], filepath: Path):
    """Pretty print validation results."""

    print(f"\n{'=' * 70}")
    print(f"Validation: {filepath.name}")
    print(f"{'=' * 70}\n")

    # Separate by level
    error_list = [e for e in errors if e.level == "ERROR"]
    warning_list = [e for e in errors if e.level == "WARNING"]
    info_list = [e for e in errors if e.level == "INFO"]

    if error_list:
        print("ERRORS (must fix):")
        for err in error_list:
            print(f"  {err}")
        print()

    if warning_list:
        print("WARNINGS (should review):")
        for err in warning_list:
            print(f"  {err}")
        print()

    if info_list:
        print("INFO:")
        for err in info_list:
            print(f"  {err}")
        print()

    # Summary
    if error_list:
        print(f"Status: ❌ FAILED ({len(error_list)} errors)")
        return 1
    elif warning_list:
        print(f"Status: ⚠️  PASSED with warnings ({len(warning_list)} warnings)")
        return 0
    else:
        print(f"Status: ✅ PASSED")
        return 0


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_triton.py <triton_file>")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    errors = validate_triton_kernel(filepath)
    exit_code = print_validation_results(errors, filepath)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
