{ lib }:

path:
let
  readToml = path: builtins.fromTOML (builtins.readFile path);
  validate =
    buildToml:
    assert lib.assertMsg ((buildToml.general.edition or null) == 6) ''
      build.toml must use edition 6, update it with:
            nix run github:huggingface/kernels#kernel-builder -- update-build'';
    buildToml;

  # Edition 6 tags kernels with `language = "<language>-<backend>"`.
  kernelLanguage = kernel: lib.head (lib.splitString "-" kernel.language);
  kernelBackend = kernel: lib.last (lib.splitString "-" kernel.language);

  toml = validate (readToml (path + "/build.toml"));

  # Torch stable ABI version for a backend, or null if it does not use the stable
  # ABI. `stable-abi` is either a single version string (all backends) or a
  # per-backend mapping.
  torchStableAbiVersionForBackend =
    backend:
    let
      stableAbi = lib.attrByPath [ "torch" "stable-abi" ] null toml;
    in
    if builtins.isString stableAbi then stableAbi else stableAbi.${backend} or null;
in
{
  inherit kernelBackend kernelLanguage toml;

  # Is the kernel a Torch kernel.
  isTorch = toml ? torch;

  # Is the kernel a tvm-ffi kernel.
  isTvmFfi = toml ? tvm-ffi;

  inherit torchStableAbiVersionForBackend;

  # Does the given backend use the torch stable ABI.
  isTorchStableAbiForBackend = backend: torchStableAbiVersionForBackend backend != null;

  # The given Torch version can build for this kernel's ABI version.
  torchCoversStableAbi =
    backend: torchVersion:
    let
      stableAbiVersion = torchStableAbiVersionForBackend backend;
    in
    stableAbiVersion != null && lib.versionAtLeast torchVersion stableAbiVersion;

  # Kernel backends.
  backends =
    let
      init = {
        cpu = false;
        cuda = false;
        metal = false;
        rocm = false;
        tpu = false;
        xpu = false;
      };
    in
    lib.foldl (backends: backend: backends // { ${backend} = true; }) init (toml.general.backends);

  # Backends for which a (compiled) kernel component is provided.
  kernelBackends =
    let
      kernels = lib.attrValues (toml.kernel or { });
      init = {
        cpu = false;
        cuda = false;
        metal = false;
        rocm = false;
        tpu = false;
        xpu = false;
      };
    in
    lib.foldl (backends: kernel: backends // { ${kernelBackend kernel} = true; }) init kernels;

  name = toml.general.name;
}
