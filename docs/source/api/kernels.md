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

`KernelDependency(repo_id, version)` identifies a kernel by its repository ID
and a [`KernelVersion`](#kernelversion). Its `repo_id` and `version` attributes
are read-only.

### KernelVersion

Use `KernelVersion.Version(number)` for a numbered kernel version or
`KernelVersion.Revision(ref)` for a Git branch, tag, or commit.

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

`KernelPaths(paths)` wraps a mapping from `KernelDependency` objects to local
paths. It behaves as a read-only mapping and accepts path strings or path-like
objects when constructed. Use `KernelPaths.from_json(text)` to read a serialized
mapping and `paths.to_json()` to write one.

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

`kernels.metadata.Metadata` represents the parsed `metadata.json` for a kernel
build variant. Use `Metadata.read_from_file(path)` to read a file or
`Metadata.from_bytes(data)` to parse JSON bytes. Both raise `ValueError` on
invalid metadata. The same class is also available as `kernels.Metadata`.

### LoadedKernel

[[autodoc]] kernels.LoadedKernel

### RepoInfo

[[autodoc]] kernels.RepoInfo
