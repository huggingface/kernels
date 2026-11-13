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
  # Validate that a Torch version entry from `versions.nix` or its
  # system-flattened version has all required attributes and no
  # unknown attributes. Returns the entry if it is valid.
  validateTorchVersion' =
    systemAttr: version:
    let
      required = [
        systemAttr
        "torchVersion"
      ];
      optional = [
        "bundleBuild"
        "cpu"
        "cudaVersion"
        "metal"
        "mlxVersion"
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

  # Validate a nested Torch version entry (with `systems`).
  validateTorchVersion = validateTorchVersion' "systems";

  # Validate a flattened Torch version entry (with `system`).
  validateFlatTorchVersion = validateTorchVersion' "system";

  # Validate that there is only one system-flattened Torch version maps
  # to a particula tvm-ffi build variant. E.g. if Torch 2.11 and 2.12
  # both have tvm-ffi enabled for CUDA 13.0, this check will fail.
  validateUniqueTvmFfiVariants =
    versions:
    let
      variantName =
        version:
        (import ./variants/tvm-ffi.nix {
          inherit lib;
          buildConfig = version // {
            backend = backend version;
          };
        }).arch;
      # Group all tvm-ffi versions by their variant name.
      byVariant = lib.groupBy variantName (builtins.filter (version: version ? tvmFfiVersion) versions);
      duplicates = lib.filterAttrs (_: group: builtins.length group > 1) byVariant;
      describe =
        variant: group:
        "${variant} (Torch ${lib.concatMapStringsSep ", " (version: version.torchVersion) group})";
    in
    lib.throwIf (duplicates != { })
      "Multiple Torch versions enable tvm-ffi for the same variant: ${lib.concatStringsSep "; " (lib.mapAttrsToList describe duplicates)}"
      versions;

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
