{
  lib,
  pkgs,
  stdenv,
  torch,
}:

let
  pythonDeps =
    let
      depsJson = builtins.fromJSON (builtins.readFile ../../kernels-data/src/python_dependencies.json);
      # Map the Nix package names to actual Nix packages.
      updatePackage = _name: dep: dep // { nix = map (pkg: pkgs.python3.pkgs.${pkg}) dep.nix; };
      updateBackend = _backend: backendDeps: lib.mapAttrs updatePackage backendDeps;
    in
    depsJson
    // {
      general = lib.mapAttrs updatePackage depsJson.general;
      backends = lib.mapAttrs updateBackend depsJson.backends;
    };

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
  resolvePythonDeps = deps: lib.flatten (map getPythonDep deps);
  resolveBackendPythonDeps = backend: deps: lib.flatten (map (getBackendPythonDep backend) deps);
}
