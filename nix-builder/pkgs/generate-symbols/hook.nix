{
  config,
  lib,
  makeSetupHook,

  generate-symbols,
  proot,
  python3,

  kernels ? python3.pkgs.kernels,

  # rpaths are stripped from kernels to make them portable, but that
  # also means that in a Nix environment the CUDA dependencies cannot
  # be located anymore. This argument is used to provide additional
  # library directories to be provided to the dynamic loader.
  libraryPath ? "",
}:

let
  useFakeSys = config.tpuSupport or false;
in
makeSetupHook {
  name = "generate-symbols-hook";
  substitutions = {
    python3 = "${python3}/bin/python";
    pythonPath = python3.pkgs.makePythonPath [ (generate-symbols.override { inherit kernels; }) ];
    inherit libraryPath;
    proot = lib.optionalString useFakeSys "${proot}/bin/proot";
    useFakeSys = lib.optionalString useFakeSys "1";
  };
} ./generate-symbols-hook.sh
