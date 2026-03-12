# This script creates the necessary files for a new kernel example in the specified directory.
#
# Example Usage:
# $ uv run scripts/init-kernel.py relu
#
# Created directory: relu
#
#   relu/
#     ├── relu_kernel/
#     │   └── relu.cu
#     ├── tests/
#     │   ├── __init__.py
#     │   └── test_relu.py
#     ├── torch-ext/
#     │   ├── relu/
#     │   │   └── __init__.py
#     │   ├── torch_binding.cpp
#     │   └── torch_binding.h
#     ├── build.toml
#     └── flake.nix
#
# ✓ Success! All files for the ReLU example have been created successfully.
#
# Next steps:
#   1. Build the kernel: cd relu && git add . && nix develop -L
#   2. Run the tests: pytest -vv tests/

import os
import argparse
import pathlib


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    GREY = "\033[90m"


def create_file_with_content(file_path: str, content: str):
    """Creates a file at 'file_path' with the specified content."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, "w") as f:
        f.write(content)


# Generate a tree view of the created files
def print_tree(directory: str, prefix: str = ""):
    entries = sorted(os.listdir(directory))

    # Process directories first, then files
    dirs = [e for e in entries if os.path.isdir(os.path.join(directory, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(directory, e))]

    # Process all items except the last one
    count = len(dirs) + len(files)

    # Print directories
    for i, dirname in enumerate(dirs):
        is_last_dir = i == len(dirs) - 1 and len(files) == 0
        connector = "└── " if is_last_dir else "├── "
        print(
            f"    {prefix}{connector}{Colors.BOLD}{Colors.BLUE}{dirname}/{Colors.ENDC}"
        )

        # Prepare the prefix for the next level
        next_prefix = prefix + ("    " if is_last_dir else "│   ")
        print_tree(os.path.join(directory, dirname), next_prefix)

    # Print files
    for i, filename in enumerate(files):
        is_last = i == len(files) - 1
        connector = "└── " if is_last else "├── "
        file_color = ""

        print(f"    {prefix}{connector}{file_color}{filename}{Colors.ENDC}")


def main():
    # Get the directory where this script is located
    script_dir = pathlib.Path(__file__).parent.resolve().parent.resolve()

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Create ReLU example files in the specified directory"
    )
    parser.add_argument(
        "target_dir", help="Target directory where files will be created"
    )
    args = parser.parse_args()

    # Get the target directory from arguments
    target_dir = args.target_dir

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(
            f"\n{Colors.CYAN}{Colors.BOLD}Created directory: {Colors.BOLD}{target_dir}{Colors.ENDC}\n"
        )
    else:
        print(
            f"\n{Colors.CYAN}{Colors.BOLD}Directory already exists: {Colors.BOLD}{target_dir}{Colors.ENDC}\n"
        )

    # get files from examples/relu
    relu_dir = script_dir / "examples" / "relu"
    for root, _, files in os.walk(relu_dir):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                content = f.read()

                # Replace kernel-builder.url with path:../ in flake.nix
                if file_path.endswith("flake.nix"):
                    kernel_builder_url_start = content.find("kernel-builder.url =")
                    kernel_builder_url_end = content.find(";", kernel_builder_url_start)
                    content = (
                        content[:kernel_builder_url_start]
                        + 'kernel-builder.url = "path:../"'
                        + content[kernel_builder_url_end:]
                    )

                target_file = file_path.replace(str(relu_dir), target_dir)
                create_file_with_content(target_file, content)

    print(f"  {Colors.BOLD}{target_dir}/{Colors.ENDC}")
    print_tree(target_dir)

    print(
        f"\n{Colors.GREEN}{Colors.BOLD}✓ Success!{Colors.ENDC} All files for the ReLU example have been created successfully."
    )

    print(f"\n{Colors.CYAN}{Colors.BOLD}Next steps:{Colors.ENDC}")

    commands = [
        "nix run nixpkgs#cachix -- use huggingface",
        f"cd {target_dir}",
        "git add .",
        "nix develop -L",
    ]

    for index, command in enumerate(commands, start=1):
        print(
            f"  {Colors.YELLOW}{index}.{Colors.ENDC} {Colors.BOLD}{command}{Colors.ENDC}"
        )

    print(
        f"\none line build:\n{Colors.GREY}{Colors.BOLD}{' && '.join(commands)}{Colors.ENDC}{Colors.ENDC}"
    )

    print(f"\n{Colors.CYAN}{Colors.BOLD}Run the tests{Colors.ENDC}")
    print(
        f"  {Colors.YELLOW}{1}.{Colors.ENDC} {Colors.BOLD}pytest -vv tests/{Colors.ENDC}"
    )

    print("")


if __name__ == "__main__":
    main()
