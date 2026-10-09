{
  config,
  lib,
  makeSetupHook,
  stdenv,

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
  generate-symbols = python3.pkgs.generate-symbols.override { inherit kernels; };
in
makeSetupHook {
  name = "get-kernel-check-hook";
  substitutions = {
    python3 = "${python3}/bin/python";
    kernels = "${python3.pkgs.makePythonPath [
      kernels
      generate-symbols
    ]}";
    inherit libraryPath;
    proot = lib.optionalString useFakeSys "${proot}/bin/proot";
    pyhook = ./get-kernel-check-hook.py;
    useFakeSys = lib.optionalString useFakeSys "1";
  };
} ./get-kernel-check-hook.sh
