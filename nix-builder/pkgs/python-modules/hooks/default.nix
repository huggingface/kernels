self: dontUse:
with self;
let
  inherit (python) pythonOnBuildForHost;
  pythonInterpreter = pythonOnBuildForHost.interpreter;
  pythonSitePackages = python.sitePackages;
  pythonCheckInterpreter = python.interpreter;
in
{
  # Ideally we'd just call this pythonImportsCheckHook, but it  would cause
  # a rebuild of most nixpkgs dependencies.
  pythonKernelImportsCheckHook = callPackage (
    { makePythonHook }:
    makePythonHook {
      name = "python-kernel-imports-check-hook.sh";
      substitutions = {
        inherit pythonCheckInterpreter pythonSitePackages;
        inherit (pkgs) proot;
      };
    } ./python-kernel-imports-check-hook.sh
  ) { };

  pythonRelaxWheelDepsHook = callPackage (
    { makePythonHook, wheel }:
    makePythonHook {
      name = "python-relax-wheel-deps-hook";
      substitutions = {
        inherit pythonSitePackages;
      };
    } ./python-relax-wheel-deps-hook.sh
  ) { };

  pythonWheelDepsCheckHook = callPackage (
    { makePythonHook, packaging }:
    makePythonHook {
      name = "python-wheel-deps-check-hook";
      propagatedBuildInputs = [ packaging ];
      substitutions = {
        inherit pythonInterpreter pythonSitePackages;
        hook = ./python-wheel-deps-check-hook.py;
      };
    } ./python-wheel-deps-check-hook.sh
  ) { };
}
