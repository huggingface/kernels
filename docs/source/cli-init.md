# kernels init

Use `kernels init` to initialize a new kernel project.

## Usage

```bash
kernels init <owner>/<repo> [--backends <backend...>] [--overwrite]
```

This creates a new directory in the current working directory with the (normalized) repo name.

## What It Does

- Downloads a project template and replaces placeholders in file names, paths, and file contents
- Optionally restricts enabled backends (updates `build.toml` and removes unused backend folders)
- Initializes a Git repo and stages the files (`git init`, `git add .`)

## Examples

Initialize a new kernel project (defaults to `metal` on macOS, `cuda` on Linux/Windows):

```bash
kernels init my-user/my-kernel
```

Enable multiple backends:

```bash
kernels init my-user/my-kernel --backends cuda cpu
```

Enable all supported backends:

```bash
kernels init my-user/my-kernel --backends all
```

Overwrite an existing directory if it exists:

```bash
kernels init my-user/my-kernel --overwrite
```

## Next Steps

`kernels init` prints suggested next steps after creating the project. A typical flow looks like:

```bash
cd my_kernel
cachix use huggingface
nix run -L --max-jobs 1 --cores 8 .#build-and-copy
uv run example.py
```

## Notes

- The `<repo>` part is normalized to lowercase, and `-` is replaced with `_`. For example, `my-user/My-Kernel` becomes a directory named `my_kernel` and a repo id `my-user/my_kernel`.
- `--backends` can be one of: `cpu`, `cuda`, `metal`, `rocm`, `xpu`, `npu`, or `all`.
- If the target directory already exists and is not empty, `kernels init` exits with an error unless `--overwrite` is set.
- The project is initialized as a Git repo (via `git init`) because Nix flakes require it.
