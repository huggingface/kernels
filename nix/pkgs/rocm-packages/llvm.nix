{
  gcc13Stdenv,
  wrapBintoolsWith,
  wrapCCWith,
  glibc,
}:
final: prev:
let
  llvm = final.rocm-llvm;
  bintools-unwrapped = final.callPackage ./bintools-unwrapped.nix {
    inherit llvm;
  };
  bintools = wrapBintoolsWith {
    bintools = bintools-unwrapped;
    libc = glibc;
    # TODO: move to unwrapped bintools
    isLLVM = true;

    extraBuildCommands = ''
      wrap ld.lld ${./ld-wrapper.sh} ${bintools-unwrapped}/bin/ld.lld
    '';
  };
  clang = final.callPackage ./clang.nix {
    inherit bintools llvm;
    stdenv = gcc13Stdenv;
  };
in
{
  llvm = {
    inherit bintools-unwrapped;
    inherit bintools;
    inherit clang;
  };
}
