# IDE setup with direnv and the kernel devshell

## Introduction

Language servers do not interpret `build.toml`, so IDE completion for
CUDA, ROCm, framework headers, and the kernel's Python wrapper does not
work out of the box. This guide shows how to configure VS Code so that
completion resolves against the same toolchain `kernel-builder`
uses.

The setup has three pieces:

- `kernel-builder create-pyproject` to emit CMake and setuptools files
  the IDE can read (see [Local Development](./local-dev.md)).
- The kernel-builder devshell, which provides the toolchain (CUDA, ROCm,
  Torch headers, etc.) from the Nix store.
- `direnv` to activate the devshell on `cd`, so VS Code inherits the
  environment through the shell.

Pinning the toolchain through Nix keeps IDE completion aligned with
the build. It also makes switching between CUDA, ROCm, or XPU a
one-line change in `.envrc`.

## Installing direnv and nix-direnv

On non-NixOS systems, install both via `nix profile`:

```bash
$ nix profile install nixpkgs#nix-direnv
```

Add the direnv hook to your shell rc (`~/.bashrc` or
`~/.zshrc`, for example):

```bash
eval "$(direnv hook bash)"    # or: direnv hook zsh
```

Source the rc file (or open a new shell) so the hook is
active in the current session:

```bash
$ source ~/.bashrc            # or: source ~/.zshrc
```

Wire `nix-direnv` into direnv:

```bash
$ mkdir -p ~/.config/direnv
$ echo 'source $HOME/.nix-profile/share/nix-direnv/direnvrc' \
    >> ~/.config/direnv/direnvrc
```

On [NixOS](https://github.com/nix-community/nix-direnv#via-system-configuration-on-nixos)
or with [home-manager](https://github.com/nix-community/nix-direnv#via-home-manager),
enable `programs.direnv` with
`nix-direnv` instead. See
[`terraform/nixos-configuration.nix`](https://github.com/huggingface/kernels/tree/main/terraform/nixos-configuration.nix)
for a working example.

## Activating the devshell with direnv

From the kernel root directory (the one containing `flake.nix` and
`build.toml`), tell direnv to use the flake's default devshell:

```bash
$ echo 'use flake' > .envrc
$ direnv allow
```

`direnv` now activates the default devshell whenever you `cd` into the
project. The devshell's `shellHook` creates and activates a `.venv` on
first entry. Confirm it picked up the toolchain and venv:

```bash
$ which nvcc
/nix/store/.../bin/nvcc
$ ls -ld .venv
drwxr-xr-x ... .venv
$ which python
/path/to/kernel/.venv/bin/python
```

If `.venv` is missing, re-run `direnv reload` and check the output for
the `Creating new venv environment in path: './.venv'` line from the
`shellHook`.

To pin a non-default build variant, name it explicitly:

```bash
$ echo 'use flake .#devShells.torch211-cxx11-rocm71-x86_64-linux' > .envrc
$ direnv allow
```

See [Build Variants](./build-variants.md) for the variant list.

## Generating IDE-facing project files

direnv puts the toolchain on `PATH`, but the C++ language server still
needs a CMake-derived `compile_commands.json` to resolve per-file
include paths. Generate the CMake/setuptools project and the file:

```bash
$ kernel-builder create-pyproject -f
$ cmake -B build-ext -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
$ ln -sf build-ext/compile_commands.json compile_commands.json
```

`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` is required: the generated CMake
does not set it. The symlink lets the language server find the file
at the project root.

As noted in [Local Development](./local-dev.md), do not commit the
generated files.

## Configuring VS Code

Open the project from a direnv-activated shell:

```bash
$ cd path/to/kernel
$ code .
```

VS Code inherits the devshell environment through the shell. No
direnv extension is needed.

Install one of the following first-party extensions for C++/CUDA
completion:

- `llvm-vs-code-extensions.vscode-clangd` (recommended for CUDA).
- `ms-vscode.cpptools` (Microsoft C/C++).

Add `.vscode/settings.json` (do not commit):

```jsonc
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",

  // clangd
  "clangd.arguments": ["--compile-commands-dir=${workspaceFolder}"],

  // Microsoft C/C++ extension
  "C_Cpp.default.compileCommands": "${workspaceFolder}/compile_commands.json"
}
```

Depending on the extension being used, the configuration above behaves
differently:

- With `clangd`, the `clangd.arguments` line is optional. clangd already
  looks in the parent directories of each source file for
  `compile_commands.json` and will find the workspace-root symlink on its
  own ([clangd docs](https://clangd.llvm.org/installation#project-setup)).
  Setting it explicitly does no harm.
- With the Microsoft C/C++ extension, the `C_Cpp.default.compileCommands`
  line is required. The extension does not pick up
  `compile_commands.json` from the workspace root on its own, unless
  another extension (such as CMake Tools) tells it where to look.

To verify, open `torch-ext/torch_binding.cpp` and hover an
`#include <torch/torch.h>` directive. The resolved path should point
into `/nix/store/...`, not a system path.

## Remote development

Use the VS Code Remote-SSH extension and put the direnv hook in the
remote shell's rc. The remote integrated terminal activates the
devshell on `cd`, and VS Code's language servers — which run on the
remote — inherit that environment. The
[`terraform/`](https://github.com/huggingface/kernels/tree/main/terraform)
setup is already configured this way.

## Switching toolchains

Change the `use flake` line in `.envrc` to point at a different
variant. For example:

```bash
# CUDA 13.0
use flake .#devShells.torch211-cxx11-cu130-x86_64-linux

# ROCm 7.1
use flake .#devShells.torch211-cxx11-rocm71-x86_64-linux

# XPU
use flake .#devShells.torch211-cxx11-xpu20253-x86_64-linux
```

Remove `.venv/` first if it was created against a different variant,
then reload direnv to recreate it via the new devshell's `shellHook`:

```bash
$ rm -rf .venv
$ direnv reload
```

## noarch kernels

For Python-only (noarch) kernels, skip the CMake step in "Generating
IDE-facing project files" and the C++ portions of the VS Code
configuration. The `direnv` setup and `python.defaultInterpreterPath`
are all that is needed.
