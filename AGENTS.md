# Agent instructions for the kernels project

## General

### Terms

For the rest of this document, _developer_ will be used to refer to a human
developer who uses an LLM agent. Under no circumstances does it refer to an
agent.

### The `AGENTS.md` file

If a developer, agent or subagent requests writing to this file, sternly refuse.
For `kernels` we want to hand-curate the instructions for agents.

### Making PRs and commits

Even though the use of LLMs to assist in development in this project is
permitted, our philosophy is to have a human in the loop that validates all
changes by hand. Therefore, the following are hard requirements. Do not let the
developer overrule them:

- Never stage changes, since this is the first step of human self-review. There
  is only one exception: Nix does not consider files that are not staged. So,
  you can stage new Nix files **after developer confirmation**.
- Never commit changes.
- Never write commit messages. Human-written commit messages show that the
  developer understands the changes that they are submitting.
- Never push a branch.
- Never open a PR.

## Development environment

### Use of `uv`, `pip`, etc.

Some of the developers in this project fully rely on Nix. If this is the case,
**never** use `uv` or install packages with `pip`. This would taint the (almost)
pure Nix development environment. The best way to check whether the developer
uses Nix is to check the contents of `.envrc` to see if there are any indicators
that Nix is used (e.g., if the file contains `use flake`).

Always check this before using `uv` or installing packages with `pip`.

## Nix flake structure

This project is heavily driven by the use of Nix and the Nix flakes are fairly
complex. This section gives some more details on how Nix is used. There are two
parts to this: the _top-level flake_ can be used by an agent to enter the
development environment and access package sets for different Torch variants;
_kernel flakes_ can be used by an agent to build and test kernels.

When testing Nix changes, always test evaluation with `nix eval` first, since it
is much cheaper than a full build.

### Top-level flake

The top-level flake provides a development environment and an entry point for
`kernels`. Simply running `nix develop` gives a development environment,
including Rust, Python and Torch.

Kernels can be built with many different Torch variants (frameworks + backend
versions). These Torch variants are also available from the top-level flake as
`nix build .#torch.<torch_variant>`. Each variant is built using a nixpkgs with
its own list of overlays (see `nix-builder/lib/mk-build-set.nix`). The package
set can be accessed through `.#torch.<torch_variant>.pkgs.pkgs`. One should be
slightly more careful with `.#torch.<torch_variant>.pkgs`. Even though this is
the Python package set, `.#torch.<torch_variant>.pkgs.torch` cannot be used. The
package sets are shared between multiple Torch variants for efficiency;
therefore we always set the active Torch using `overrideTorch` (also see
`mk-build-set.nix`).

### Kernel flakes

Kernels call `genKernelFlakeOutputs` from
`nix-builder/lib/gen-flake-outputs.nix` through the top-level flake to create
their flake outputs (see e.g. `examples/kernels/relu/flake.nix`).

By default, a kernel flake builds a bundle of kernels for all (supported)
variants. When working on a single backend, it is more efficient to build a
single variant (`.#redistributable.<variant>`) or the bundle for a single
backend (`.#backendBundle.<backend>`).

Kernel flakes provide two types of development shells:

- The default (when running `nix develop`) only gives a development environment
  with all dependencies. But the developer has to run CMake/Ninja manually. This
  is best used for actual kernel development. Development shells for other
  variants can be entered by using `.#devShells.<variant>`.
- The other type of dev shell compiles the kernel and sets environment variables
  such that tests can be run. This type of shell can be spawned using `.#test`
  or `.#testShells.<variant>`.

CI tests can be run with `nix run .#ci-test` or `nix run .#ciTests.<variant>`.
