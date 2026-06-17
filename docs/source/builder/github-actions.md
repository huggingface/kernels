# Building and testing kernels with GitHub Actions

Compiling a kernel is CPU-intensive and testing it requires an accelerator (such as a GPU), two things that GitHub's standard runners do not provide cheaply. Instead of maintaining self-hosted runners, you can offload both steps to [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) directly from a GitHub Actions workflow.

Two prebuilt actions make this possible:

- [`huggingface/kernel-builder-job`](https://github.com/huggingface/kernel-builder-job) runs the Nix kernel builder on a CPU flavor and uploads the result to the Hub.
- [`huggingface/hf-jobs-action`](https://github.com/huggingface/hf-jobs-action) runs an arbitrary script on any flavor (including GPUs), which is convenient for testing the kernel you just built.

A typical setup has two workflows: one that **builds** the kernel on push, and one that **tests** it on a GPU. They communicate through the Hub, the build uploads artifacts, the test pulls them back down.

## Prerequisites

- An [HF access token](https://huggingface.co/settings/tokens) with the `job.write` permission. If the test job loads gated models, the token also needs read access to them.
- The token stored as a repository secret named `HF_TOKEN` (**Settings → Secrets and variables → Actions**).
- A kernel repository on the Hub to upload to, with kernel-creation access for the owning user or org (see [Building kernels](build.md#uploading-your-kernel-to-the-hub)).

> [!NOTE]
> Jobs run under the `namespace` you specify (your username or an org) and
> count against that namespace's compute quota.

## Building on push

The build action checks out nothing by itself, your script clones the exact commit and invokes the Nix builder. Compilation happens on the HF Jobs CPU flavor, and `build-and-upload` pushes the finished variants to the Hub.

```yaml
# .github/workflows/build-kernel.yml
name: Build Kernel

on:
  push:
    branches: [main]
    paths:
      - "csrc/**"
      - "torch-ext/**"
      - flake.nix
      - flake.lock
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build kernel via HF Jobs
        uses: huggingface/kernel-builder-job@main
        with:
          token: ${{ secrets.HF_TOKEN }}
          namespace: your-username
          flavor: cpu-xl
          timeout: "21600"
          script: |
            # The container starts with `set -x`; disable tracing so the
            # token below is not echoed into the streamed logs.
            set +x
            export HF_TOKEN="${{ secrets.HF_TOKEN }}"
            # Rebuild artifacts, so skip pulling existing LFS blobs.
            export GIT_LFS_SKIP_SMUDGE=1

            git clone "${{ github.server_url }}/${{ github.repository }}" kernel
            cd kernel
            git checkout "${{ github.sha }}"

            nix run github:huggingface/kernels#kernel-builder -- build-and-upload \
              --max-jobs 4 \
              --cores 8 \
              --repo-id your-username/your-kernel
```

The path filter keeps the build from running on unrelated commits, and `workflow_dispatch` lets you trigger it by hand from the **Actions** tab. The upload destination is taken from `--repo-id` (or, if omitted, from the `repo-id`/`version` fields in `build.toml`).

> [!TIP]
> Builds can take a long time on the first run because every PyTorch and CUDA
> variant is compiled. Set a generous `timeout` (the example allows six hours)
> and rely on the [Hugging Face binary cache](build.md#using-the-hugging-face-binary-cache)
> to keep subsequent builds fast.

### `kernel-builder-job` inputs

| Input       | Required | Default            | Description                                       |
| ----------- | -------- | ------------------ | ------------------------------------------------- |
| `token`     | yes      |                    | HF token with `job.write` permission.             |
| `namespace` | yes      |                    | HF namespace (username or org) that owns the job. |
| `script`    | yes      |                    | Shell script to run in the container.             |
| `flavor`    | no       | `cpu-upgrade`      | Hardware flavor (e.g. `cpu-xl`).                  |
| `image`     | no       | Nix + cachix image | Container image to run the build in.              |
| `timeout`   | no       | `1200`             | Maximum seconds to wait for the job.              |

The action exposes `job_id` and `job_url` outputs that link to the run on huggingface.co.

## Testing on a GPU

Once the kernel is on the Hub, the generic jobs action runs a test script on a GPU flavor. The `files` input copies repository files into the container (under `/tmp/files` by default), and a [`uv` script](https://docs.astral.sh/uv/guides/scripts/) with inline dependencies keeps the environment self-contained.

```yaml
# .github/workflows/run-tests.yml
name: Run tests

on:
  push:
    branches: [main]
    paths:
      - scripts/test.py
  workflow_dispatch:

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run test.py on an HF Jobs GPU
        uses: huggingface/hf-jobs-action@main
        with:
          token: ${{ secrets.HF_TOKEN }}
          namespace: your-username
          flavor: rtx-pro-6000
          image: ghcr.io/astral-sh/uv:python3.10-bookworm
          timeout: "3600"
          files: scripts/test.py
          script: |
            set +x
            export HF_TOKEN="${{ secrets.HF_TOKEN }}"
            uv run /tmp/files/test.py
```

The test script pulls the kernel straight from the Hub with the [`kernels`](../basic-usage.md) library, so it always runs against the artifacts the build workflow just published:

```python
# scripts/test.py
# /// script
# dependencies = ["kernels", "torch"]
# ///
from kernels import get_kernel

kernel = get_kernel("your-username/your-kernel")
# ... exercise the kernel and assert on the results ...
```

Run the build workflow before the test workflow so the Hub has a fresh kernel to pull. For tightly coupled steps, you can also have one workflow trigger the other, or combine both jobs in a single workflow with a `needs:` dependency.

### `hf-jobs-action` inputs

| Input        | Required | Default      | Description                                             |
| ------------ | -------- | ------------ | ------------------------------------------------------- |
| `token`      | yes      |              | HF token with `job.write` permission.                   |
| `namespace`  | yes      |              | HF namespace (username or org) that owns the job.       |
| `image`      | yes      |              | Container image to run.                                 |
| `script`     | yes      |              | Shell script to execute in the container.               |
| `flavor`     | no       | `cpu-basic`  | Hardware flavor (e.g. `rtx-pro-6000`).                  |
| `files`      | no       |              | Newline-separated repo files to copy into the job.      |
| `files_dest` | no       | `/tmp/files` | Directory the files are copied to inside the container. |
| `env`        | no       | `{}`         | Environment variables as a JSON object.                 |
| `timeout`    | no       | `1200`       | Maximum seconds to wait for the job.                    |

## Choosing a flavor

Flavors map to the machine types available on Hugging Face Jobs, CPU flavors such as `cpu-upgrade` and `cpu-xl` for builds, and GPU flavors such as `l4x1`, `a100-large`, `h200`, or `rtx-pro-6000` for tests. Pick the most reasonable GPU that fits your model to keep jobs low cost. The current list and pricing are in the [Hugging Face Jobs documentation](https://huggingface.co/docs/huggingface_hub/guides/jobs).

> [!WARNING]
> HF Jobs containers start with shell tracing enabled (`set -x`). Always run
> `set +x` before exporting `HF_TOKEN` so the token does not leak into the
> streamed build logs.
