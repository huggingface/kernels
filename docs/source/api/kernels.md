# Kernels API Reference

## Main Functions

### get_kernel

[[autodoc]] kernels.get_kernel

### get_local_kernel

[[autodoc]] kernels.get_local_kernel

### has_kernel

[[autodoc]] kernels.has_kernel

### get_kernel_variants

[[autodoc]] kernels.get_kernel_variants

### get_loaded_kernels

[[autodoc]] kernels.get_loaded_kernels

## Loading locked kernels

### load_kernel

[[autodoc]] kernels.load_kernel

### get_locked_kernel

[[autodoc]] kernels.get_locked_kernel

## Loading with a custom resolver

### kernels.load.get_kernel_with_resolver

[[autodoc]] kernels.load.get_kernel_with_resolver

### kernels.hf_hub.get_hf_api

[[autodoc]] kernels.hf_hub.get_hf_api

## Resolving kernel dependencies

Import these types from `kernels.resolve`.

### KernelDependency

[[autodoc]] kernels.resolve.KernelDependency

### KernelVersion

[[autodoc]] kernels.resolve.KernelVersion

#### KernelVersion.Version

[[autodoc]] kernels.resolve.KernelVersion.Version

#### KernelVersion.Revision

[[autodoc]] kernels.resolve.KernelVersion.Revision

### Resolver

Implement `resolve(api=..., backend=..., kernel=...)` to return a
[`~kernels.resolve.LocalKernel`] or [`~kernels.resolve.RemoteKernel`]. Return
`None` when the resolver cannot resolve the dependency.

[[autodoc]] kernels.resolve.Resolver

### SequentialResolver

[[autodoc]] kernels.resolve.SequentialResolver

### RepoPathsResolver

[[autodoc]] kernels.resolve.RepoPathsResolver

### KernelPathsResolver

[[autodoc]] kernels.resolve.KernelPathsResolver

### KernelPaths

[[autodoc]] kernels.resolve.KernelPaths

### LocalKernel

[[autodoc]] kernels.resolve.LocalKernel

### RemoteKernel

[[autodoc]] kernels.resolve.RemoteKernel

## Validating kernels

Import these protocols, combinators, and default validator factories from
`kernels.validate`.

### MetadataValidator

Implement `validate_metadata(metadata=..., variant=...)` to check each build
variant’s metadata before installation.

[[autodoc]] kernels.validate.MetadataValidator

### KernelValidator

Implement `validate_kernel(kernel=...)` to check each installed
[`~kernels.resolve.LocalKernel`] before import.

[[autodoc]] kernels.validate.KernelValidator

### AllMetadataValidator

[[autodoc]] kernels.validate.AllMetadataValidator

### AllKernelValidator

[[autodoc]] kernels.validate.AllKernelValidator

### default_metadata_validators

[[autodoc]] kernels.validate.default_metadata_validators

### default_kernel_validators

[[autodoc]] kernels.validate.default_kernel_validators

## Classes

### Metadata

Import `Metadata` from `kernels.metadata` or directly from `kernels`.

[[autodoc]] kernels.metadata.Metadata

### LoadedKernel

[[autodoc]] kernels.LoadedKernel

### RepoInfo

[[autodoc]] kernels.RepoInfo
