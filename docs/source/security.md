# Security model

This page describes what the Kernel Hub and the `kernels` Python package do from a **security** perspective: what runs on your machine, what guarantees you should **not** assume, and how publisher checks work today.

If you **publish** kernels, see also [Security for kernel builders](builder/security.md) (supply-chain hygiene, PR review, reproducible builds).

## Executive summary

Loading a kernel executes **native machine code** (shared libraries) and **Python code** from the kernel package with the same privileges as your application process.

By default, [`get_kernel`](basic-usage.md) only allows Hub repositories whose owning **organization** is on a fixed client-side allowlist. Other repositories raise unless you explicitly opt in with `trust_remote_code=True`, which is equivalent to agreeing to run **unreviewed code** from that repository.

Some other APIs query or install kernels **without** repeating that publisher gate; they are still safe only when **you** already restricted which repositories and revisions are used (see [Which APIs enforce publisher checks](#which-apis-enforce-publisher-checks)).

## What executes when you load a kernel

When a kernel is imported (typically via [`get_kernel`](basic-usage.md), [`get_local_kernel`](basic-usage.md), [`get_locked_kernel`](locking.md), or [`load_kernel`](locking.md)):

1. **Artifacts are resolved** from the Hub cache or a local tree (via `huggingface_hub` when downloading).
2. **`metadata.json`** is read to determine the Python module layout and declared Python dependencies.
3. **Declared Python dependencies** are validated against an allowlist enforced by `kernels` (see [Kernel requirements — Python dependencies](kernel-requirements.md)).
4. **Native libraries** shipped inside the chosen build variant are loaded by the dynamic linker as needed when Python bindings run.
5. **Kernel Python entrypoints** (`__init__.py` and related modules) are executed with normal import semantics.

There is **no sandbox**: kernel code runs as your user and can perform any operation your process can perform (file access subject to OS permissions, GPU execution, network calls from Python or native code, and so on).

## Threat model and responsibilities

| Concern | Responsibility |
| --- | --- |
| Choosing **which** repositories and revisions to run | Application authors and end users |
| **Publisher gate** on direct Hub loads via [`get_kernel`](basic-usage.md) | `kernels` (see below) |
| **Curated Python dependency names** for kernels | `kernels` metadata validation |
| **Reviewing** merged source and build outputs before publishing | Kernel maintainers ([builder security](builder/security.md)) |
| **Hub account security**, tokens, CI secrets | Users and organization admins |

The `kernels` package does **not** cryptographically attest builds by default, does **not** vet native binaries beyond Hub transport and your lockfile hashes (when locking is used), and does **not** confine execution.

## Publisher checks in [`get_kernel`](basic-usage.md)

[`get_kernel`](basic-usage.md) performs a **client-side** check before downloading or importing from the Hub:

- With the default `trust_remote_code=False`, loading succeeds only if the repository ID’s owning organization (the segment before the first `/`) is one of the following: **`kernels-community`**, **`kernels-staging`**, **`kernels-test`**, **`sglang`**.
- With `trust_remote_code=True`, that gate is **skipped** for any `kernel`-type repository you request.

Repositories under personal namespaces (`username/repo`) are **not** on the default allowlist unless your username exactly matches one of the names above (which is unlikely). Such kernels require an explicit opt-in.

### Signing identities (`trust_remote_code=[...]`)

The API accepts `trust_remote_code` as a list of strings for future **signing identity** verification. **This is not implemented yet.** Passing a list currently emits a warning and **does not** replace the default publisher check; behavior falls through to the same organization allowlist as `False`.

Until signing is implemented, treat a list value like `False`: only allowlisted organizations load without error, unless you pass `trust_remote_code=True`.

### Hub metadata and badges

The Hugging Face Hub may expose publisher-trust signals (for example metadata or UI badges). Those signals inform discovery and policy on the Hub side. **`kernels` does not yet substitute Hub-side trust metadata for the client allowlist described above** when enforcing `get_kernel` defaults.

## Which APIs enforce publisher checks

| API | Publisher allowlist on Hub loads |
| --- | --- |
| [`get_kernel`](basic-usage.md) | **Yes** (unless `trust_remote_code=True`) |
| [`LayerRepository`](layers.md) / [`FuncRepository`](layers.md), Hub-backed `load()` | **Yes** — delegates to [`get_kernel`](basic-usage.md) with that repository’s `trust_remote_code` setting |
| [`has_kernel`](basic-usage.md) | **No** (Hub metadata probe only; does not import the kernel) |
| `install_kernel`, `install_kernel_all_variants` | **No** |
| [`kernels download`](cli-download.md) | **No** |
| [`get_locked_kernel`](locking.md), [`load_kernel`](locking.md) (`kernels.utils`) | **No** — these paths download/import without repeating the publisher gate; trust depends entirely on what was locked and who controls `kernels.lock` |
| [`LockedLayerRepository`](layers.md) / [`LockedFuncRepository`](layers.md), Hub `load()` | **Yes** — resolves the revision from a lockfile, then calls [`get_kernel`](basic-usage.md) (unless `trust_remote_code=True`) |
| [`get_local_kernel`](basic-usage.md) | **No** (loads whatever tree you point at) |

Because of this split, **lockfiles and local overrides must only pin repositories you trust.** A malicious or compromised dependency could add a `kernels.lock` entry pointing at an arbitrary Hub revision; importing that kernel still runs its code.

Note the distinction between **`kernels.utils` helpers** and **locked Hub repositories in the mapping API**: [`get_locked_kernel`](locking.md) and [`load_kernel`](locking.md) download via internal install paths that **skip** the publisher check, whereas [`LockedLayerRepository`](layers.md) / [`LockedFuncRepository`](layers.md) ultimately call [`get_kernel`](basic-usage.md) and **apply** the same gate as a direct Hub load.

[`LocalLayerRepository`](layers.md) and other **local** mapping helpers call [`get_local_kernel`](basic-usage.md); they never apply the Hub publisher allowlist, since they execute whatever tree you provide.

Mitigations:

- Review **`kernels.lock`** (and related packaging metadata) in code review.
- Prefer **`get_kernel`** with default trust settings for ad hoc experimentation from the Hub.
- Pin **`revision` / `version`** explicitly rather than floating branches where policy requires stability ([locking guide](locking.md)).

## Tokens and Hub access

Downloading private kernels or raising Hub rate limits requires credentials configured for [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/index) (for example the `HF_TOKEN` environment variable).

Treat tokens like passwords: scope them minimally, rotate them if leaked, and never commit them to source control.

## Telemetry

Library identification may be sent as part of Hub requests (user-agent metadata). See [How can I disable kernel reporting in the user-agent?](faq.md#how-can-i-disable-kernel-reporting-in-the-user-agent) and the Hugging Face Hub documentation on [environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

## Reporting security issues

If you believe you have found a **security vulnerability** in `kernels`,
kernel-builder, or Hub kernel handling, please email
[**security@huggingface.co**](mailto:security@huggingface.co). Someone from the Hugging Face
security team can advise on appropriate disclosure steps.

Where GitHub private vulnerability reporting is enabled for this repository,
you can also use the **Security** tab on [github.com/huggingface/kernels](https://github.com/huggingface/kernels)
(**Report a vulnerability**).

Avoid using **public GitHub issues alone** for sensitive vulnerability reports. Public issues are still appropriate for documentation improvements or non-sensitive hardening discussions.

## See also

- [Security for kernel builders](builder/security.md)
- [Kernel requirements — Trusted publishers](kernel-requirements.md)
- [Environment variables](env.md)
