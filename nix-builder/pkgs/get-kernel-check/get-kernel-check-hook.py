import os
from pathlib import Path

from kernels.deps import AllValidator, default_validators
from kernels.hf_hub import _get_hf_api
from kernels.load import get_kernel_with_resolver
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

variant = os.getenv("variant")
if not variant:
    raise ValueError("`variant` environment variable is not set by Nix derivation")

kernel = KernelDependency(repo_id=out, version=KernelVersion.Version(0))

# The build host may not expose the accelerator being targeted (e.g. Metal is
# undetectable inside the sandboxed macOS build), so point the resolver at the
# variant the derivation is expected to produce instead of relying on backend
# auto-detection. This also fails the check if the build produced a different
# variant than expected.
variant_path = Path(out) / variant
if not variant_path.is_dir():
    raise FileNotFoundError(
        f"Expected build variant `{variant}` is not present in `{out}`"
    )

resolvers = [
    RepoPathsResolver(local_kernels={out: variant_path}),
    KernelPathsResolver(kernel_paths=kernel_paths),
]

get_kernel_with_resolver(
    api=_get_hf_api(),
    backend=None,
    kernel=kernel,
    resolver=SequentialResolver(resolvers=resolvers),
    validator=AllValidator(validators=default_validators()),
)
