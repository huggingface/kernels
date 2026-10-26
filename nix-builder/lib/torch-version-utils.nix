{ lib }:
let
  isCpu = version: version.cpu or false;
  isCuda = version: version ? cudaVersion;
  isMetal = version: version.metal or false;
  isRocm = version: version ? rocmVersion;
  isTpu = version: version.tpu or false;
  isXpu = version: version ? xpuVersion;

in
rec {
  # Validate that a Torch version entry from `versions.nix` has all required
  # attributes and no unknown attributes. Returns the entry if it is valid.
  validateTorchVersion =
    version:
    let
      required = [
        "systems"
        "torchVersion"
      ];
      optional = [
        "bundleBuild"
        "cpu"
        "cudaVersion"
        "metal"
        "ptxasVersion"
        "rocmVersion"
        "tpu"
        "tvmFfiVersion"
        "xpuVersion"
      ];
      missingAttrs = lib.filter (attr: !(version ? ${attr})) required;
      unknownAttrs = lib.subtractLists (required ++ optional) (builtins.attrNames version);
      context = builtins.toJSON version;
    in
    lib.throwIf (missingAttrs != [ ])
      "Torch version is missing required attribute(s) ${lib.concatStringsSep ", " missingAttrs}: ${context}"
      (
        lib.throwIf (unknownAttrs != [ ])
          "Torch version has unknown attribute(s) ${lib.concatStringsSep ", " unknownAttrs}: ${context}"
          version
      );

  # Expand { systems = [ a b ]; .. } to [ { system = a; ..} { system = b; .. } ]
  flattenSystems =
    versions:
    lib.foldl' (
      acc: version:
      acc
      ++ map (system: (builtins.removeAttrs version [ "systems" ]) // { inherit system; }) version.systems
    ) [ ] (map validateTorchVersion versions);

  backend =
    version:
    if isCpu version then
      "cpu"
    else if isCuda version then
      "cuda"
    else if isMetal version then
      "metal"
    else if isRocm version then
      "rocm"
    else if isTpu version then
      "tpu"
    else if isXpu version then
      "xpu"
    else
      throw "Could not find compute framework: no CUDA, ROCm, XPU version specified and CPU and Metal are not enabled";
}
