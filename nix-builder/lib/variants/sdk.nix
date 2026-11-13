{
  lib,
  buildConfig,
}:

# SDK build variants. Shape:
#
# - `sdk-<backend>[-<version>]-cpython-<python_version>-<abi>-<platform>-<os>`

let
  # Backends that have a version in the variant string.
  backendVersion =
    {
      cuda = buildConfig.cudaVersion or null;
      rocm = buildConfig.rocmVersion or null;
      xpu = buildConfig.xpuVersion or null;
    }
    .${buildConfig.backend} or null;

  backendString =
    if backendVersion == null then
      buildConfig.backend
    else
      "${buildConfig.backend}-${lib.versions.majorMinor backendVersion}";
in
{
  # The variant string for an SDK with the given Python version and ABI,
  # e.g. `{ version = "3.12"; abi = "abi3"; }`.
  variant =
    {
      version,
      abi,
    }:
    "sdk-${backendString}-cpython-${lib.versions.majorMinor version}-${abi}-${buildConfig.system}";
}
