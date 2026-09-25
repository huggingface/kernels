# Kernels security


`kernels` downloads and loads remote code, so you need clear trust boundaries. We aim to address the following attack vectors:

- A kernel developer ships malicious code inside a kernel.
- The Hub credentials of a kernel developer are compromised and an
  attacker uses them to add malicious code to a kernel.

We do not aim to address the following attack vectors:

- Local Hub cache manipulation if the manipulation is done by the attacker
  on the system that loads the kernels. The reasoning here is that the
  attacker could do anything at this point, including tampering with the
  software that loads the kernel.

This document discusses the security measures that `kernels` implements
to address the first two attack vectors. Improving kernel security is an
ongoing effort, and we welcome feedback and contributions.

## Malicious kernel developers

As a `kernels` user, ask two trust questions before you load a kernel:

- Do I trust the developer?
- Do I trust the developer to properly secure their infrastructure and
  credentials?

kernels also blocks accidental loads from publishers you have not opted into, such as from trusted publishers, the [kernels-community](https://huggingface.co/kernels-community), and kernel provenance.

### Trusted publishers

By default, a call such as

```python
activation = get_kernel("someorg/activation", version=1)
```

will raise an exception unless `someorg` is a trusted publisher. A
trusted publisher is a kernel developer who is deemed reputable by the
Hugging Face kernels team. The trusted publisher status is queried from
the Hub when downloading a kernel.

The set of trusted publishers is intentionally very small and generally
only contains organizations that have years of experience in developing
kernels and maintaining secure build infrastructure.

Pass `trust_remote_code=True` to allow any publisher, or pass a list of repo IDs to allow only those repose (plus trusted publishers). The default is `False` to always block unknown publishers.

```python
activation = get_kernel(
    "someorg/activation",
    version=1,
    trust_remote_code=["someorg/activation"]
)
```

If your library loads kernels on behalf of its users, expose a flag so they can control untrusted publishers. For example:

```python
TRUST_REMOTE_KERNELS = os.environ.get("MYLIB_TRUST_REMOTE_KERNELS", "0") == "1"

activation = get_kernel(
    "someorg/activation",
    version=1,
    trust_remote_code=["someorg/activation"] if TRUST_REMOTE_KERNELS else False
)
```

### kernels-community

The [kernels-community](https://huggingface.co/kernels-community/kernels)
organization provides a large number of kernels that are maintained by
Hugging Face. Kernels from this organization are publicly maintained
through [GitHub](https://github.com/huggingface/kernels-community) and
are built using ephemeral build containers. PRs that are merged into
kernels-community go through an automated security audit and can only
be merged by a very small group of maintainers.

The kernels-community organization is a trusted publisher, so it
provides good kernel coverage out of the box.

### Kernel provenance

You can also check that a build came from the claimed source. Every kernel built with kernel-builder records provenance in the build variant's `metadata.json`, including the Git commits of:

- The revision of kernel-builder that the kernel was built with.
- The revision of the kernel source itself.

This information is stored in the build variant's `metadata.json`. For
example:

```json
"provenance": {
  "kernel-builder": {
    "version": "0.17.0-dev0",
    "commit": "a7f0afdb29a6a3372b1d47180cc0c182454c5e3b",
    "dirty": false
  },
  "kernel": {
    "commit": "a137a8498a30a98631a3deedfa44ea01bf1cef7c",
    "dirty": false
  }
}
```

For a full example, see the [flash-attn3 kernel](https://huggingface.co/kernels/kernels-community/flash-attn3/blob/v2/build/torch-stable-abi29-cu126-x86_64-linux/metadata.json).

This provenance information can be used to rebuild the kernel at the exact
same revisions of the kernel and the builder. A binary diff between the
remote build and your local build can help reveal any tampering.

Kernel builds are largely reproducible because kernels are built using Nix
inside a sandbox. `kernel-builder` pins the full toolchain (compiler, C library, etc.) in `flake.lock`. You can find these pins in the
[kernels](https://github.com/huggingface/kernels) repository.

## Kernel compromises


Even if you trust a kernel developer, their credentials might be
compromised. An attacker could use the credentials to upload a malicious
version of a kernel.`kernels` offer two protections against these attack vectors:

- Kernel locking
- Code signing


### Kernel locking

Kernel locking records the Git commit hash of a kernel in a `kernels.lock`
file. After the kernel is locked, load it with `get_locked_kernel` (downloads if needed) or `load_kernel (requires a pre-downloaded kernel and errors if it is missing). Either path only loads the locked commit. If an attacker compromises a kernel Hub repository and pushes a new, malicious version, it will not be used. See [Lock kernel versions](./locking) for how to lock the kernels of a project.

An attacker can circumvent the lock by trying to craft a commit
that collides with the SHA-1 hash. However, this is currently hard,
since the Hugging Face Hub platform uses a Git implementation with
SHA-1 collision detection ([sha1dc](https://github.com/cr-marcstevens/sha1collisiondetection)).

### Code signing

`kernels` can verify kernels with cosign. 

On load, `kernels` checks that the files match signed digests in `metadata.json`. Signing uses cosign with short-lived keys, and the signature is recorded in a [ledger](https://docs.sigstore.dev/logging/overview/). That combination makes leaked CI signing keys much harder to reuse.

The builder computes the SHA-256 digest of each file in the kernel and stores it in `metadata.json`:

```json
"digest": {
  "algorithm": "sha256",
  "files": {
    "__init__.py": "iY6XPtdZUgS8HizpndYTF+U/5kUmeLsADwCNM0UXMME=",
    "_ops.py": "5ecEPtZkkJLggvzM/luMYUmHIpCZXiBMEwXG0GmIMBU=",
    "_rmsnorm_xpu_89d4054.abi3.so": "yb8gSiUShkgjnrdDNTPJ0prhLPQAvTIFWwbllgqK5po=",
    "layers.py": "8KMmxy30Olm/16XW7vvHAwWNVWcnA8js0nNuoOSGEO8="
  }
}
```

The digests are used to verify that the kernel files were not tampered
with. Of course, the hashes need to be protected against tampering as well.
To this end, `metadata.json` can be signed using cosign. The cosign tool
signs using ephemeral keys that are only valid for a short time. Moreover,
the signing is recorded in a tamper-proof, immutable
[ledger](https://docs.sigstore.dev/logging/overview/). Together, ephemeral
keys plus the ledger prevent reuse of private keys that were leaked in CI
attacks.

Aside from the main signature, cosign also records information about how
the signature was made, such as the OIDC issuer, the source repository, and
the workflow path/branch.

Signature verification performs
the following steps:

- Verify the signature against the given policy. The default policy only accepts kernels signed by workflows in the `huggingface/kernel-community` GitHub repository.
- Verify the authenticity of `metadata.json` using the signature.
- Use the digests in `metadata.json` to verify the kernel files.

At this time, a signature verification error will only result in a warning.
Moreover, signature verification is only performed when the `sigstore` Python
package is installed. However, we will make signature verification mandatory
in the future.

The same steps can be performed on demand with the
[`kernels verify-signature`](cli-verify-signature.md) command.

#### Signature verification receipts

To avoid the high cost of signature verification, a kernel is only verified
in full once. The first time a kernel is loaded, we perform all the steps
above. Upon successful verification, we write a receipt file to the kernels
cache.

When a receipt is found on a later load, the signature and digest checks are
skipped. The signing certificate is still checked against the policy, since
the receipt could have been written by a verification with a different
policy.

Receipts are stored by kernel identity. The name of a receipt file is a hash
of:

- The repo ID
- The Git commit hash
- The build variant

This means that downloading a different revision or build variant of the
same kernel will not collide with the existing receipt and will be verified as
expected.
