import os
from pathlib import Path

from kernels.load import get_kernel_with_resolver
from kernels.hf_hub import _get_hf_api
from kernels.resolver import KernelPathsResolver, RepoPathsResolver, SequentialResolver
from kernels_data import KernelDependency, KernelPaths, KernelVersion

out = os.getenv("out")
if not out:
    raise ValueError("`out` environment variable is not set by Nix derivation")

kernelDeps = os.getenv("kernelDeps")
if not kernelDeps:
    raise ValueError("`kernelDeps` environment variable is not set by Nix derivation")


with open(kernelDeps) as f:
    kernel_paths = KernelPaths.from_json(f.read())

kernel = KernelDependency(repo_id=out, version=KernelVersion.Version(0))

# The build host may not expose the accelerator being targeted (e.g. Metal is
# undetectable inside the sandboxed macOS build), so point the resolver at the
# derivation's single variant instead of relying on backend auto-detection.
[variant] = (p.parent for p in Path(out).glob("*/metadata.json"))

resolvers = [
    RepoPathsResolver(local_kernels={out: variant}),
    KernelPathsResolver(kernel_paths=kernel_paths),
]

get_kernel_with_resolver(
    api=_get_hf_api(),
    backend=None,
    kernel=kernel,
    resolver=SequentialResolver(resolvers=resolvers),
)
