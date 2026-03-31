# Onboarding Script Plan: `curl | bash` with Nix-first Approach

## Goal

Provide a single command that gets a new kernel developer from zero to a working
`kernel-builder` installation:

```bash
curl -fsSL https://raw.githubusercontent.com/huggingface/kernels/main/install.sh | bash
```

The key insight: install Determinate Nix first, then install `kernel-builder`
through Nix via `nix profile install`. This avoids requiring users to have Rust
installed and sidesteps Rust toolchain version issues entirely.

---

## Architecture

```
install.sh (entry point)
    |
    +--> 1. Check/install Nix (Determinate installer)
    +--> 2. Configure binary cache (huggingface cachix)
    +--> 3. Install kernel-builder via nix profile
    +--> 4. Print next-steps summary
```

---

## Implementation Steps

### Step 1: Create `install.sh` at repo root

A POSIX-compatible shell script with the following sections:

#### Prerequisites

`kernel-builder` is already exposed per-system at
`packages.<system>.kernel-builder` in the root `flake.nix` (line 208), so
`nix profile install github:huggingface/kernels#kernel-builder` already works.
The flake declares the `huggingface.cachix.org` substituter in `nixConfig`, so
the binary is fetched from cache when Nix trusts the flake's nixConfig
(Determinate Nix does this by default). If cache misses are an issue, consider
also setting `packages.<system>.default = kernel-builder` so that
`nix profile install github:huggingface/kernels` (without the fragment) works
too.

#### 1a. Preamble & environment detection

- `set -euo pipefail`
- Detect OS (`uname -s`) and architecture (`uname -m`)
- Define color helpers for terminal output (with `NO_COLOR` support)
- Bail out on unsupported platforms (e.g., Windows/WSL detection with a pointer
  to WSL-specific instructions if needed)
- Supported targets: `x86_64-linux`, `aarch64-linux`, `aarch64-darwin`
- **macOS**: check for `xcode-select -p` and **warn** (not error) if Xcode is
  not installed — it's required for Metal kernels but not for the install itself

#### 1b. Check if Nix is already installed

- Check for `nix` in `PATH` or the Determinate Nix default locations
  (`/nix/var/nix/profiles/default/bin/nix`)
- If found: print version, skip to step 1d
- If not found: proceed to step 1c

#### 1c. Install Determinate Nix

- Print a clear message: "Installing Determinate Nix..."
- Run the Determinate installer:
  ```bash
  curl -fsSL https://install.determinate.systems/nix | sh -s -- install --no-confirm
  ```
- The `--no-confirm` flag avoids a second interactive prompt (the user already
  opted in by running our script). Consider making this configurable via an
  environment variable (e.g., `KERNEL_BUILDER_INTERACTIVE=1` to drop
  `--no-confirm`).
- Source the Nix profile so `nix` is available in the current shell:
  ```bash
  . /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
  ```
- Verify `nix --version` succeeds after install

#### 1d. Configure the Hugging Face binary cache

This is critical for usability — without the cache, users will build PyTorch
from source (hours of compilation).

- Check if the `huggingface.cachix.org` substituter is already configured
  (parse `nix show-config` output)
- If not configured, run:
  ```bash
  nix run nixpkgs#cachix -- use huggingface
  ```
- This adds the cache to the Nix config without requiring `cachix` to be
  permanently installed

#### 1e. Install kernel-builder via Nix profile

```bash
nix profile install github:huggingface/kernels#kernel-builder
```

This pulls the pre-built `kernel-builder` binary from the Hugging Face cache
(no Rust compilation needed). It also installs shell completions (bash/fish/zsh)
that are already set up in the Nix derivation.

- Verify `kernel-builder --version` succeeds after install

#### 1f. Print success message and next steps

```
kernel-builder installed successfully!

Next steps:
  1. Create a new kernel:     kernel-builder init my-kernel
  2. Build your kernel:       cd my-kernel && nix run .#build-and-copy -L
  3. Read the docs:           https://huggingface.co/docs/kernels/

To update kernel-builder later:
  nix profile upgrade --all

Note: you may need to restart your shell or run:
  . /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
```

---

### Step 2: Update documentation

- **`docs/source/builder/writing-kernels.md`**: The primary landing page for
  kernel authors. Update the "Setting up environment" section (currently only
  mentions Terraform) to lead with the `curl | bash` one-liner as the
  recommended way to get started.
- **`docs/source/builder/nix.md`**: Keep as the detailed Nix reference. Add a
  note at the top that the install script handles all of this automatically,
  linking back to `writing-kernels.md`.
- Keep the existing manual instructions in `nix.md` as-is for users who prefer
  step-by-step control.

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `install.sh` | **Create** | The main `curl \| bash` onboarding script |
| `docs/source/builder/writing-kernels.md` | **Edit** | Add install one-liner to "Setting up environment" |
| `docs/source/builder/nix.md` | **Edit** | Add note that install script automates these steps |

---

## Open Questions

1. **`--no-confirm` for Determinate installer** — Should we always pass this, or
   ask the user? The Determinate installer's own prompt is informative (shows
   what it will do). Passing `--no-confirm` is more frictionless but less
   transparent.

2. ~~**Pinning the flake ref**~~ **Resolved**: use `main` as the install ref
   (`github:huggingface/kernels#kernel-builder`). `main` is defined as always
   pointing to the latest release — i.e., tags are cut from `main`.

3. ~~**Updating kernel-builder**~~ **Resolved**: document the upgrade command
   in the success message and in `writing-kernels.md`. No `--upgrade` flag in
   the script — just tell users to run:
   ```
   nix profile upgrade --all
   ```

4. **Flake nixConfig trust** — When a user runs `nix profile install` on a flake
   with `nixConfig.extra-substituters`, Nix may prompt them to trust it (unless
   using Determinate Nix which trusts by default). If using the official Nix
   installer on Linux, we may need to configure the cache explicitly before
   the `nix profile install` step. The script handles this in step 1d.
