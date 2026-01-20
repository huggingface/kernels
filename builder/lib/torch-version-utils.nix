{ lib }:
let
  isCpu = version: version.cpu or false;
  isCuda = version: version ? cudaVersion;
  isMetal = version: version.metal or false;
  isRocm = version: version ? rocmVersion;
  isXpu = version: version ? xpuVersion;

in
rec {
  # Expand { systems = [ a b ]; .. } to [ { system = a; ..} { system = b; .. } ]
  flattenSystems = lib.foldl' (
    acc: version:
    acc
    ++ map (system: (builtins.removeAttrs version [ "systems" ]) // { inherit system; }) version.systems
  ) [ ];

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
    else if isXpu version then
      "xpu"
    else
      throw "Could not find compute framework: no CUDA, ROCm, XPU version specified and CPU and Metal are not enabled";
}
