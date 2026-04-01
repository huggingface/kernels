{ lib }:

path:
let
  readToml = path: builtins.fromTOML (builtins.readFile path);
  validate =
    buildToml:
    let
      hasBackends = buildToml.general ? backends;
      kernels = lib.attrValues (buildToml.kernel or { });

    in
    assert lib.assertMsg hasBackends ''
      build.toml seems to be of an older version, update it with:
            nix run github:huggingface/kernels#kernel-builder update-build build.toml'';
    buildToml;
  toml = validate (readToml (path + "/build.toml"));
in
{
  inherit toml;

  # Is the kernel a Torch kernel.
  isTorch = toml ? torch;

  # Is the kernel a tvm-ffi kernel.
  isTvmFfi = toml ? tvm-ffi;

  # Kernel backends.
  backends =
    let
      init = {
        cpu = false;
        cuda = false;
        metal = false;
        rocm = false;
        xpu = false;
      };
    in
    lib.foldl (backends: backend: backends // { ${backend} = true; }) init (toml.general.backends);

  # Backends for which a (compiled) kernel component is provided.
  kernelBackends =
    let
      kernels = lib.attrValues (toml.kernel or { });
      kernelBackend = kernel: kernel.backend;
      init = {
        cpu = false;
        cuda = false;
        metal = false;
        rocm = false;
        xpu = false;
      };
    in
    lib.foldl (backends: kernel: backends // { ${kernelBackend kernel} = true; }) init kernels;

  name = toml.general.name;
}
