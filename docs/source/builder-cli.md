# CLI reference for kernel-builder

This document contains the help content for the `kernel-builder` command-line program.

**Command Overview:**

* [`kernel-builder`↴](#kernel-builder)
* [`kernel-builder completions`↴](#kernel-builder-completions)
* [`kernel-builder init`↴](#kernel-builder-init)
* [`kernel-builder build`↴](#kernel-builder-build)
* [`kernel-builder build-and-copy`↴](#kernel-builder-build-and-copy)
* [`kernel-builder build-and-upload`↴](#kernel-builder-build-and-upload)
* [`kernel-builder upload`↴](#kernel-builder-upload)
* [`kernel-builder create-pyproject`↴](#kernel-builder-create-pyproject)
* [`kernel-builder devshell`↴](#kernel-builder-devshell)
* [`kernel-builder list-variants`↴](#kernel-builder-list-variants)
* [`kernel-builder testshell`↴](#kernel-builder-testshell)
* [`kernel-builder update-build`↴](#kernel-builder-update-build)
* [`kernel-builder validate`↴](#kernel-builder-validate)
* [`kernel-builder skills`↴](#kernel-builder-skills)
* [`kernel-builder skills add`↴](#kernel-builder-skills-add)
* [`kernel-builder clean-pyproject`↴](#kernel-builder-clean-pyproject)

## `kernel-builder`

Build Hugging Face Hub kernels

**Usage:** `kernel-builder <COMMAND>`

###### **Subcommands:**

* `completions` — Generate shell completions
* `init` — Initialize a new kernel project from template
* `build` — Build the kernel locally (alias for build-and-copy)
* `build-and-copy` — Build the kernel and copy artifacts locally
* `build-and-upload` — Build the kernel and upload to Hugging Face Hub
* `upload` — Upload kernel build artifacts to the Hugging Face Hub
* `create-pyproject` — Generate CMake files for a kernel extension build
* `devshell` — Spawn a kernel development shell
* `list-variants` — List build variants
* `testshell` — Spawn a kernel test shell
* `update-build` — Update a `build.toml` to the current format
* `validate` — Validate the build.toml file
* `skills` — Install skills for AI coding assistants (Claude, Codex, OpenCode)
* `clean-pyproject` — Clean generated artifacts



## `kernel-builder completions`

Generate shell completions

**Usage:** `kernel-builder completions <SHELL>`

###### **Arguments:**

* `<SHELL>`

  Possible values: `bash`, `elvish`, `fish`, `powershell`, `zsh`




## `kernel-builder init`

Initialize a new kernel project from template

**Usage:** `kernel-builder init [OPTIONS] [PATH]`

###### **Arguments:**

* `<PATH>` — Directory to initialize (defaults to current directory)

###### **Options:**

* `--name <OWNER/REPO>` — Name of the kernel repo (e.g. `drbh/my-kernel`)
* `--backends <BACKENDS>` — Backends to enable (`all`, `cpu`, `cuda`, `metal`, `neuron`, `rocm`, `xpu`)

* `--overwrite` — Overwrite existing scaffold files (preserves other files)



## `kernel-builder build`

Build the kernel locally (alias for build-and-copy)

**Usage:** `kernel-builder build [OPTIONS] [KERNEL_DIR]`

###### **Arguments:**

* `<KERNEL_DIR>` — Directory of the kernel project (defaults to current directory)

###### **Options:**

* `--variant <VARIANT>` — Build a specific variant
* `--max-jobs <MAX_JOBS>` — Maximum number of parallel Nix build jobs
* `--cores <CORES>` — Number of CPU cores to use for each build job
* `-L`, `--print-build-logs` — Print full build logs on standard error



## `kernel-builder build-and-copy`

Build the kernel and copy artifacts locally

**Usage:** `kernel-builder build-and-copy [OPTIONS] [KERNEL_DIR]`

###### **Arguments:**

* `<KERNEL_DIR>` — Directory of the kernel project (defaults to current directory)

###### **Options:**

* `--max-jobs <MAX_JOBS>` — Maximum number of parallel Nix build jobs
* `--cores <CORES>` — Number of CPU cores to use for each build job
* `-L`, `--print-build-logs` — Print full build logs on standard error



## `kernel-builder build-and-upload`

Build the kernel and upload to Hugging Face Hub

**Usage:** `kernel-builder build-and-upload [OPTIONS] [KERNEL_DIR]`

###### **Arguments:**

* `<KERNEL_DIR>` — Directory of the kernel project (defaults to current directory)

###### **Options:**

* `--variant <VARIANT>` — Build a specific variant
* `--max-jobs <MAX_JOBS>` — Maximum number of parallel Nix build jobs
* `--cores <CORES>` — Number of CPU cores to use for each build job
* `-L`, `--print-build-logs` — Print full build logs on standard error
* `--repo-id <REPO_ID>` — Repository ID on the Hugging Face Hub (e.g. `user/my-kernel`)
* `--branch <BRANCH>` — Upload to a specific branch (defaults to `v{version}` from metadata)
* `--private` — Create the repository as private
* `--repo-type <REPO_TYPE>` — Repository type on Hugging Face Hub (`kernel` by default, or `model` for legacy repos)

  Default value: `kernel`

  Possible values: `model`, `kernel`




## `kernel-builder upload`

Upload kernel build artifacts to the Hugging Face Hub

**Usage:** `kernel-builder upload [OPTIONS] [KERNEL_DIR]`

###### **Arguments:**

* `<KERNEL_DIR>` — Directory of the kernel build (defaults to current directory)

###### **Options:**

* `--repo-id <REPO_ID>` — Repository ID on the Hugging Face Hub (e.g. `user/my-kernel`). Defaults to `general.hub.repo-id` from `build.toml`
* `--branch <BRANCH>` — Upload to a specific branch (defaults to `v{version}` from metadata)
* `--private` — Create the repository as private
* `--repo-type <REPO_TYPE>` — Repository type on Hugging Face Hub (`kernel` by default, or `model` for legacy repos)

  Default value: `kernel`

  Possible values: `model`, `kernel`




## `kernel-builder create-pyproject`

Generate CMake files for a kernel extension build

**Usage:** `kernel-builder create-pyproject [OPTIONS] [KERNEL_DIR] [TARGET_DIR]`

###### **Arguments:**

* `<KERNEL_DIR>`
* `<TARGET_DIR>` — The directory to write the generated files to (directory of `BUILD_TOML` when absent)

###### **Options:**

* `-f`, `--force` — Force-overwrite existing files
* `--ops-id <OPS_ID>` — This is an optional unique identifier that is suffixed to the kernel name to avoid name collisions. (e.g. Git SHA)



## `kernel-builder devshell`

Spawn a kernel development shell

**Usage:** `kernel-builder devshell [OPTIONS] [KERNEL_DIR]`

###### **Arguments:**

* `<KERNEL_DIR>`

###### **Options:**

* `--variant <VARIANT>` — Use a specific variant
* `--max-jobs <MAX_JOBS>` — Maximum number of parallel Nix build jobs
* `--cores <CORES>` — Number of CPU cores to use for each build job
* `-L`, `--print-build-logs` — Print full build logs on standard error



## `kernel-builder list-variants`

List build variants

**Usage:** `kernel-builder list-variants [OPTIONS] [KERNEL_DIR]`

###### **Arguments:**

* `<KERNEL_DIR>`

###### **Options:**

* `--arch` — Only list variants for the current architecture



## `kernel-builder testshell`

Spawn a kernel test shell

**Usage:** `kernel-builder testshell [OPTIONS] [KERNEL_DIR]`

###### **Arguments:**

* `<KERNEL_DIR>`

###### **Options:**

* `--variant <VARIANT>` — Use a specific variant
* `--max-jobs <MAX_JOBS>` — Maximum number of parallel Nix build jobs
* `--cores <CORES>` — Number of CPU cores to use for each build job
* `-L`, `--print-build-logs` — Print full build logs on standard error



## `kernel-builder update-build`

Update a `build.toml` to the current format

**Usage:** `kernel-builder update-build [KERNEL_DIR]`

###### **Arguments:**

* `<KERNEL_DIR>`



## `kernel-builder validate`

Validate the build.toml file

**Usage:** `kernel-builder validate [KERNEL_DIR]`

###### **Arguments:**

* `<KERNEL_DIR>`



## `kernel-builder skills`

Install skills for AI coding assistants (Claude, Codex, OpenCode)

**Usage:** `kernel-builder skills <COMMAND>`

###### **Subcommands:**

* `add` — Install a kernels skill for an AI assistant



## `kernel-builder skills add`

Install a kernels skill for an AI assistant

**Usage:** `kernel-builder skills add [OPTIONS]`

###### **Options:**

* `--skill <SKILL>` — Skill to install

  Default value: `cuda-kernels`

  Possible values: `cuda-kernels`, `rocm-kernels`

* `--claude` — Install for Claude
* `--codex` — Install for Codex
* `--opencode` — Install for OpenCode
* `-g`, `--global` — Install globally (user-level) instead of in the current project directory
* `--dest <DEST>` — Install into a custom destination (path to skills directory)
* `--force` — Overwrite existing skills in the destination



## `kernel-builder clean-pyproject`

Clean generated artifacts

**Usage:** `kernel-builder clean-pyproject [OPTIONS] [KERNEL_DIR] [TARGET_DIR]`

###### **Arguments:**

* `<KERNEL_DIR>`
* `<TARGET_DIR>` — The directory to clean from (directory of `BUILD_TOML` when absent)

###### **Options:**

* `-d`, `--dry-run` — Show what would be deleted without actually deleting
* `-f`, `--force` — Force deletion without confirmation
* `--ops-id <OPS_ID>` — This is an optional unique identifier that is suffixed to the kernel name to avoid name collisions. (e.g. Git SHA)



<hr/>

<small><i>
    This document was generated automatically by
    <a href="https://crates.io/crates/clap-markdown"><code>clap-markdown</code></a>.
</i></small>
