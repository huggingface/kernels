self: dontUse:
with self;
let
  inherit (python) pythonOnBuildForHost;
  pythonInterpreter = pythonOnBuildForHost.interpreter;
  pythonSitePackages = python.sitePackages;
in
{
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
