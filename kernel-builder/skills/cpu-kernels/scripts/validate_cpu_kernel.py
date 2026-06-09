#!/usr/bin/env python3
"""
Validate C++ CPU kernel for common issues.

Checks build.toml, C++ source files, and kernel structure for correctness
and common pitfalls.

Usage:
    python scripts/validate_cpu_kernel.py <kernel_dir>
    python scripts/validate_cpu_kernel.py .
"""

import os
import re
import sys
from pathlib import Path
from typing import List


class ValidationError:
    def __init__(self, level: str, message: str, file: str = None, line_num: int = None):
        self.level = level  # 'ERROR', 'WARNING', 'INFO'
        self.message = message
        self.file = file
        self.line_num = line_num

    def __str__(self):
        prefix = {"ERROR": "X", "WARNING": "!", "INFO": "i"}[self.level]
        loc = ""
        if self.file:
            loc += f" [{self.file}"
            if self.line_num:
                loc += f":{self.line_num}"
            loc += "]"
        return f"[{prefix}] {self.level}: {self.message}{loc}"


def validate_build_toml(kernel_dir: Path) -> List[ValidationError]:
    """Validate build.toml configuration."""
    errors = []
    build_toml = kernel_dir / "build.toml"

    if not build_toml.exists():
        errors.append(ValidationError("ERROR", "build.toml not found"))
        return errors

    with open(build_toml) as f:
        content = f.read()

    # Check for CPU backend sections
    if 'backend = "cpu"' not in content:
        errors.append(ValidationError("ERROR", "No CPU backend sections found in build.toml", "build.toml"))

    # Check for include directive
    sections = re.findall(r'\[kernel\.\w+\]', content)
    for section in sections:
        # Find the section content
        start = content.index(section)
        next_section = content.find("[kernel.", start + 1)
        section_content = content[start:next_section] if next_section != -1 else content[start:]

        if 'backend = "cpu"' in section_content and "include" not in section_content:
            errors.append(ValidationError(
                "WARNING",
                f"Section {section} missing 'include' directive for header resolution",
                "build.toml",
            ))

    # Check AVX512 compiler flags
    if "-mavx512f" in content:
        # Core flags required for all AVX512 kernels
        core_flags = ["-mavx512bf16", "-mavx512vl"]
        for flag in core_flags:
            if flag not in content:
                errors.append(ValidationError(
                    "WARNING",
                    f"AVX512 section missing core flag: {flag}",
                    "build.toml",
                ))
        # GEMM-specific flags — only check if kernel has GEMM-like names or vbmi
        gemm_indicators = ["gemm", "gptq", "quantiz", "bnb", "bitsandbytes", "megablocks", "moe"]
        is_gemm_kernel = any(ind in content.lower() for ind in gemm_indicators)
        if is_gemm_kernel:
            gemm_flags = ["-mavx512dq", "-mavx512bw", "-mavx512vbmi"]
            for flag in gemm_flags:
                if flag not in content:
                    errors.append(ValidationError(
                        "INFO",
                        f"GEMM kernel may benefit from flag: {flag}",
                        "build.toml",
                    ))
        else:
            # For non-GEMM kernels, just note that dq/bw/vbmi are optional
            pass

    # Check for -fopenmp
    if "-mavx512f" in content and "-fopenmp" not in content:
        errors.append(ValidationError(
            "WARNING",
            "AVX512 section missing -fopenmp flag",
            "build.toml",
        ))

    return errors


def validate_cpp_file(filepath: Path) -> List[ValidationError]:
    """Validate a C++ source file for common CPU kernel issues."""
    errors = []
    fname = filepath.name

    with open(filepath) as f:
        lines = f.readlines()

    source = "".join(lines)

    # 1. Check for aligned loads (should use unaligned)
    for i, line in enumerate(lines):
        if "_mm512_load_" in line and "_mm512_loadu_" not in line:
            # Exclude comments
            stripped = line.lstrip()
            if not stripped.startswith("//") and not stripped.startswith("*"):
                errors.append(ValidationError(
                    "WARNING",
                    "Using aligned load (_mm512_load_*). Prefer _mm512_loadu_* for safety.",
                    fname, i + 1,
                ))
        if "_mm256_load_" in line and "_mm256_loadu_" not in line:
            stripped = line.lstrip()
            if not stripped.startswith("//") and not stripped.startswith("*"):
                errors.append(ValidationError(
                    "WARNING",
                    "Using aligned load (_mm256_load_*). Prefer _mm256_loadu_* for safety.",
                    fname, i + 1,
                ))

    # 2. Check for missing edge/tail handling in AVX512 files
    if "avx512" in fname.lower():
        has_vector_ops = "_mm512_" in source
        has_tail = "remainder" in source.lower() or "tail" in source.lower() or "mask" in source.lower()
        if has_vector_ops and not has_tail:
            errors.append(ValidationError(
                "INFO",
                "No tail/remainder handling detected. Ensure hidden_size is always divisible by vector width, "
                "or add masked/scalar fallback.",
                fname,
            ))

    # 3. Check for cpu_features.hpp inclusion in dispatcher (direct or via header)
    if fname.endswith("_cpu.cpp") and "avx" not in fname:
        # Check for direct include or include via main header (which may transitively include cpu_features)
        has_cpu_features = "cpu_features.hpp" in source or "cpu_features" in source
        # Also check if it includes a header that might include cpu_features.hpp
        if not has_cpu_features:
            # Check included headers for transitive include
            included_headers = re.findall(r'#include\s+"([^"]+\.hpp)"', source)
            parent_dir = filepath.parent
            for header in included_headers:
                header_path = parent_dir / header
                if header_path.exists():
                    with open(header_path) as hf:
                        if "cpu_features.hpp" in hf.read():
                            has_cpu_features = True
                            break
        if not has_cpu_features:
            errors.append(ValidationError(
                "WARNING",
                "Dispatcher file should include cpu_features.hpp for runtime feature detection.",
                fname,
            ))

    # 4. Check for mixing intrinsics from different ISA levels
    has_avx2 = "_mm256_" in source
    has_avx512 = "_mm512_" in source
    if has_avx2 and has_avx512:
        # AVX512 files legitimately use _mm256 for partial-width ops (e.g., zero-point loading)
        # Only warn if this is an AVX2-only file
        is_avx512_file = "avx512" in fname.lower()
        if not is_avx512_file:
            errors.append(ValidationError(
                "WARNING",
                "Mixing AVX2 (_mm256_*) and AVX512 (_mm512_*) intrinsics in same file. "
                "Each ISA tier should be in a separate translation unit.",
                fname,
            ))

    # 5. Check for proper OpenMP usage
    if "#pragma omp" in source and "#include <omp.h>" not in source:
        # OpenMP pragmas without header — might still work but check
        pass

    # 6. Check for float64 usage (slow on CPU too)
    for i, line in enumerate(lines):
        if "double " in line and "epsilon" not in line.lower() and "eps" not in line.lower():
            stripped = line.lstrip()
            if not stripped.startswith("//") and not stripped.startswith("*"):
                errors.append(ValidationError(
                    "INFO",
                    "double (float64) detected. Consider float/bf16 for better SIMD throughput.",
                    fname, i + 1,
                ))

    return errors


def validate_torch_binding(kernel_dir: Path) -> List[ValidationError]:
    """Validate torch_binding.cpp (located at torch-ext/torch_binding.cpp)."""
    errors = []
    # Check both possible locations: torch-ext/torch_binding.cpp (standard) and root
    binding = kernel_dir / "torch-ext" / "torch_binding.cpp"
    if not binding.exists():
        binding = kernel_dir / "torch_binding.cpp"
    
    if not binding.exists():
        errors.append(ValidationError("ERROR", "torch_binding.cpp not found (expected at torch-ext/torch_binding.cpp)"))
        return errors

    # Warn if not in standard location
    if binding.parent.name != "torch-ext":
        errors.append(ValidationError(
            "WARNING",
            "torch_binding.cpp should be in torch-ext/ directory (torch-ext/torch_binding.cpp)",
            str(binding.relative_to(kernel_dir)),
        ))

    with open(binding) as f:
        source = f.read()

    if "registration.h" not in source:
        errors.append(ValidationError(
            "ERROR",
            "torch_binding.cpp should include registration.h",
            "torch_binding.cpp",
        ))

    if "TORCH_LIBRARY_EXPAND" not in source:
        errors.append(ValidationError(
            "ERROR",
            "torch_binding.cpp should use TORCH_LIBRARY_EXPAND macro for op registration",
            "torch_binding.cpp",
        ))

    if "REGISTER_EXTENSION" not in source:
        errors.append(ValidationError(
            "WARNING",
            "torch_binding.cpp should use REGISTER_EXTENSION macro",
            "torch_binding.cpp",
        ))

    return errors


def validate_kernel_structure(kernel_dir: Path) -> List[ValidationError]:
    """Validate overall kernel directory structure."""
    errors = []

    # Find CPU kernel subdirectories
    cpu_dirs = [d for d in kernel_dir.iterdir() if d.is_dir() and "cpu" in d.name.lower()]

    if not cpu_dirs:
        errors.append(ValidationError(
            "WARNING",
            "No *_cpu/ subdirectory found. CPU kernel code should be in <kernel>_cpu/.",
        ))
        return errors

    for cpu_dir in cpu_dirs:
        # Check for cpu_features.hpp
        if not (cpu_dir / "cpu_features.hpp").exists():
            errors.append(ValidationError(
                "ERROR",
                f"Missing cpu_features.hpp in {cpu_dir.name}/",
            ))

        # Check for at least one AVX512 implementation
        avx512_files = list(cpu_dir.glob("*avx512*"))
        if not avx512_files:
            errors.append(ValidationError(
                "WARNING",
                f"No AVX512 implementation found in {cpu_dir.name}/",
            ))

    return errors


def validate_kernel(kernel_dir: Path) -> List[ValidationError]:
    """Run all validation checks."""
    errors = []

    errors.extend(validate_build_toml(kernel_dir))
    errors.extend(validate_torch_binding(kernel_dir))
    errors.extend(validate_kernel_structure(kernel_dir))

    # Validate all C++ files
    for ext in ("*.cpp", "*.hpp"):
        for cpp_file in kernel_dir.rglob(ext):
            errors.extend(validate_cpp_file(cpp_file))

    return errors


def print_results(errors: List[ValidationError], kernel_dir: Path) -> int:
    """Pretty print validation results."""

    print(f"\n{'=' * 70}")
    print(f"CPU Kernel Validation: {kernel_dir}")
    print(f"{'=' * 70}\n")

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

    if error_list:
        print(f"Status: FAILED ({len(error_list)} errors)")
        return 1
    elif warning_list:
        print(f"Status: PASSED with warnings ({len(warning_list)} warnings)")
        return 0
    else:
        print(f"Status: PASSED")
        return 0


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_cpu_kernel.py <kernel_dir>")
        sys.exit(1)

    kernel_dir = Path(sys.argv[1]).resolve()
    if not kernel_dir.exists():
        print(f"Error: Directory not found: {kernel_dir}")
        sys.exit(1)

    errors = validate_kernel(kernel_dir)
    exit_code = print_results(errors, kernel_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
