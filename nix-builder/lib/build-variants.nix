{ lib }:
let
  inherit (import ./torch-version-utils.nix { inherit lib; })
    backend
    flattenSystems
    ;
in
rec {

  buildName =
    let
      inherit (import ./version-utils.nix { inherit lib; }) flattenVersion;
      computeString =
        version:
        if backend version == "cpu" then
          "cpu"
        else if backend version == "cuda" then
          "cu${flattenVersion (lib.versions.majorMinor version.cudaVersion)}"
        else if backend version == "rocm" then
          "rocm${flattenVersion (lib.versions.majorMinor version.rocmVersion)}"
        else if backend version == "metal" then
          "metal"
        else if backend version == "xpu" then
          "xpu${flattenVersion (lib.versions.majorMinor version.xpuVersion)}"
        else
          throw "No compute framework set in Torch version";
    in
    version:
    if version.system == "aarch64-darwin" then
      "torch${flattenVersion version.torchVersion}-${computeString version}-${version.system}"
    else
      "torch${flattenVersion version.torchVersion}-cxx11-${computeString version}-${version.system}";

  # Build variants included in bundle builds.
  buildVariants =
    torchVersions:
    let
      bundleBuildVersions = lib.filter (version: version.bundleBuild or false);
    in
    lib.foldl' (
      acc: version:
      let
        path = [
          version.system
          (backend version)
        ];
        pathVersions = lib.attrByPath path [ ] acc ++ [ (buildName version) ];
      in
      lib.recursiveUpdate acc (lib.setAttrByPath path pathVersions)
    ) { } (flattenSystems (bundleBuildVersions torchVersions));
}
