{
  lib,
  pkgs,
  torch,
}:

let
  cppDeps = {
    "cutlass_2_10" = [
      pkgs.cutlass_2_10
    ];
    "cutlass_3_5" = [
      pkgs.cutlass_3_5
    ];
    "cutlass_3_6" = [
      pkgs.cutlass_3_6
    ];
    "cutlass_3_8" = [
      pkgs.cutlass_3_8
    ];
    "cutlass_3_9" = [
      pkgs.cutlass_3_9
    ];
    "cutlass_4_0" = [
      pkgs.cutlass_4_0
    ];
    "torch" = [
      torch
    ];
    "cutlass_sycl" = [ torch.xpuPackages.cutlass-sycl ];
    "metal-cpp" = [
      pkgs.metal-cpp.dev
    ];
  };

  pythonDeps =
    let
      depsJson = builtins.fromJSON (builtins.readFile ../build2cmake/src/python_dependencies.json);
      # Map the Nix package names to actual Nix packages.
      updatePackage = _name: dep: dep // { nix = map (pkg: pkgs.python3.pkgs.${pkg}) dep.nix; };
      updateBackend = _backend: backendDeps: lib.mapAttrs updatePackage backendDeps;
    in
    depsJson
    // {
      general = lib.mapAttrs updatePackage depsJson.general;
      backends = lib.mapAttrs updateBackend depsJson.backends;
    };

  getCppDep = dep: cppDeps.${dep} or (throw "Unknown dependency: ${dep}");
  getPythonDep =
    dep: lib.attrByPath [ "general" dep "nix" ] (throw "Unknown Python dependency: ${dep}") pythonDeps;
  getBackendPythonDep =
    backend: dep:
    let
      backendDeps = lib.attrByPath [
        "backends"
        backend
      ] (throw "Unknown backend: ${backend}") pythonDeps;
    in
    lib.attrByPath [
      dep
      "nix"
    ] (throw "Unknown Python dependency for backend `${backend}`: ${dep}") backendDeps;
in
{
  resolveCppDeps = deps: lib.flatten (map getCppDep deps);
  resolvePythonDeps = deps: lib.flatten (map getPythonDep deps);
  resolveBackendPythonDeps = backend: deps: lib.flatten (map (getBackendPythonDep backend) deps);
}
