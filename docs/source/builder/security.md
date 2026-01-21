# Security

## Introduction

As a kernel builder, you provide code that might be run on thousands or
even millions of machines. This comes with the responsibility of ensuring
no malicious code is distributed.

Below, we provide guidelines to help avoid common attack vectors. These
are _in addition to_ common-sense security practices, such as keeping
your credentials/tokens safe, being vigilant against machine compromise,
and doing proper code reviews.

## Handling pull requests

The Hugging Face Hub allows users to submit pull requests to your
repositories. **Never** merge a pull request that contains a `build/`
directory. The binaries inside the `build/` directory might be compromised
even when the source code looks fine. When a pull request includes
`build/`, ask the submitter to re-submit it without builds. Build the
kernel on your own trusted infrastructure after the PR is merged.

When a PR does not contain build outputs and is ready to review, _carefully_
review every changed line, also taking security into account. Even if the
PR is from a trusted party, review it as if their credentials might have
been compromised.

## Build hygiene

If possible, do builds on a dedicated build machine/VM that is only used
for sandboxed builds (non-macOS kernel-builder builds are sandboxed as
well). Specialized machines are less likely to be compromised, especially
when they are accessed with a hardware-stored SSH key that requires user
interaction.

## Supporting reproducibility

Reproducible builds are very helpful to verify that no malicious code has
slipped into a kernel. If a kernel build is reproducible, then anyone can
rebuild a kernel and verify the binaries match the distributed binaries.
Full reproducibility is a goal we are working toward in `kernel-builder`.

However, this also requires assistance from the kernel builder. This section
describes what you need to do to make reproducible builds possible.

### Only build kernels with Nix sandboxing enabled.

Nix can be used with sandboxing disabled to support systems that do not
support sandboxing (e.g. Linux systems that are configured to disable
mount/network namespaces). **Never** build kernels with sandboxing disabled.
Not only can this cause stray system dependencies to be picked up, but
it can also cause other impurities to slip into the build, making it
impossible to reproduce the build. You can verify that sandboxing is enabled
using `nix-info`:

```bash
$ nix-shell -p nix-info --run "nix-info -m"
 - system: `"x86_64-linux"`
 - host os: `Linux 6.12.39, NixOS, 25.11 (Xantusia), 25.11.20250723.1744f3d`
 - multi-user?: `yes`
 - sandbox: `yes`
 - version: `nix-env (Nix) 2.28.4`
 - nixpkgs: `/nix/store/fqwc3ghi5qfdmzklpwssbamxcqj1vgl3-source`
```

### Do not build from dirty Git trees

Before building a kernel, ensure that all changes are committed. This
makes it possible to reproduce a build from exactly the same source code.
We bake the git shorthash into the ops name, so that it is clear from
which git hash a kernel was built.
