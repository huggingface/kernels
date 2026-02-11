# kernels upload

Use `kernels upload` to upload built kernels to the Hugging Face Hub.

## Usage

```bash
kernels upload <kernel_dir> --repo-id <repo_id> [--branch <branch>] [--private]
```

## What It Does

- Creates a repository on the Hub if it doesn't exist
- Uploads the kernel build artifacts from the specified directory
- If a build variant already exists in the repo, replaces the existing files

## Examples

Upload a kernel build:

```bash
kernels upload ./build --repo-id my-username/my-kernel
```

Upload to a specific branch:

```bash
kernels upload ./build --repo-id my-username/my-kernel --branch dev
```

Upload as a private repository:

```bash
kernels upload ./build --repo-id my-username/my-kernel --private
```

## Options

| Option      | Required | Description                                             |
| ----------- | -------- | ------------------------------------------------------- |
| `--repo-id` | Yes      | Repository ID on the Hub (e.g., `username/kernel-name`) |
| `--branch`  | No       | Upload to a specific branch instead of main             |
| `--private` | No       | Create the repository as private                        |

## Prerequisites

You must be authenticated with the Hugging Face Hub:

```bash
huggingface-cli login
```

## Notes

- The `kernel_dir` should contain the build output (typically the `build/` directory from your kernel project)
- If uploading a new variant to an existing repo, only that variant's files are replaced
- Make sure your kernel passes [`kernels check`](cli-check.md) before uploading

## See Also

- [kernels check](cli-check.md) - Verify kernel compliance before uploading
- [kernels init](cli-init.md) - Create a new kernel project
