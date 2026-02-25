# kernels upload

Use `kernels upload` to upload built kernels to the Hugging Face Hub.

## Usage

```bash
kernels upload <kernel_dir> --repo-id <repo_id> [--branch <branch>] [--private]
```

## What It Does

- This will take care of creating a repository on the Hub with the `repo_id` provided.
- If a repo with the `repo_id` already exists and if it contains a `build` with the build variant
  being uploaded, it will attempt to delete the files existing under it.
- Make sure to be authenticated (run `hf auth login` if not) to be able to perform uploads to the Hub.

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
