#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

_SYSTEM_LIBS: Dict[str, str] = {
    "libc++.1.dylib": "/usr/lib/libc++.1.dylib",
    "libc++.1.0.dylib": "/usr/lib/libc++.1.dylib",
}


def run_command(command: List[str], check: bool = True) -> str:
    """Executes a shell command and returns its output."""
    try:
        result = subprocess.run(
            command, check=check, capture_output=True, text=True, encoding="utf-8"
        )
        return result.stdout.strip()
    except FileNotFoundError:
        print(
            f"Error: Command '{command[0]}' not found. Is it in your PATH?",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}", file=sys.stderr)
        print(f"Output:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)


def get_dependencies(file_path: Path) -> List[str]:
    """
    Uses `otool -l` to parse the LC_LOAD_DYLIB commands and find the
    raw, un-resolved dependency paths stored in the binary.
    """
    output = run_command(["otool", "-l", str(file_path)])
    lines = output.splitlines()

    dependencies = []
    name_regex = re.compile(r"^\s+name\s+(\S+)\s+\(offset \d+\)")

    for i, line in enumerate(lines):
        if "cmd LC_LOAD_DYLIB" in line:
            # The 'name' field, which contains the path, is consistently
            # located two lines after the 'cmd' line in the otool output.
            if i + 2 < len(lines):
                name_line = lines[i + 2]
                match = name_regex.match(name_line)
                if match:
                    dependencies.append(match.group(1))

    return dependencies


def rewrite_nix_paths(file_path: Path):
    """
    Finds and rewrites Nix store paths in a binary's dependencies
    to be @rpath-relative.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Error: File not found at {file_path}")

    dependencies = get_dependencies(file_path)

    for old_path in dependencies:
        if old_path.startswith("/nix/store/"):
            lib_name = os.path.basename(old_path)

            # Hmpf, since the Big Sur dynamic linker cache, we cannot
            # simply test for the existence of system libraries. So
            # use a table instead.
            if lib_name in _SYSTEM_LIBS:
                new_path = _SYSTEM_LIBS[lib_name]
            else:
                new_path = f"@rpath/{lib_name}"

            print(f"{old_path} -> {new_path}")

            command = [
                "install_name_tool",
                "-change",
                old_path,
                new_path,
                str(file_path),
            ]
            run_command(command)


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite Nix store paths in a macOS dynamic library (.so or .dylib) to be @rpath-relative.",
    )
    parser.add_argument(
        "file", type=Path, help="Path to the .so or .dylib file to modify."
    )
    args = parser.parse_args()
    rewrite_nix_paths(args.file)


if __name__ == "__main__":
    main()
