{ lib }:

path:

let
  inherit (lib) fileset;
  kernelConfig = (import ./kernel-config.nix { inherit lib; }) path;
  nameToPath = path: name: path + "/${name}";
  kernels = kernelConfig.toml.kernel or { };
  extConfig = kernelConfig.toml.torch or kernelConfig.toml.tvm-ffi or { };
  pyExt =
    extConfig.pyext or [
      "py"
      "pyi"
    ];
  pyFilter = file: builtins.any (ext: file.hasExt ext) pyExt;
  extSrc = extConfig.src or [ ] ++ [ "build.toml" ];
  torchExtPath = path + "/torch-ext";
  tvmFfiExtPath = path + "/tvm-ffi-ext";
  pySrcSet =
    let
      path =
        if builtins.pathExists torchExtPath then
          torchExtPath
        else if builtins.pathExists tvmFfiExtPath then
          tvmFfiExtPath
        else
          throw "Kernel should have torch-ext or tvm-ffi-ext directory";
    in
    fileset.fileFilter pyFilter path;
  pyTestsSet =
    let
      testsPath = path + "/tests";
    in
    if builtins.pathExists testsPath then
      fileset.fileFilter pyFilter (path + "/tests")
    else
      fileset.empty;
  kernelsSrc = fileset.unions (
    lib.flatten (lib.mapAttrsToList (name: buildConfig: map (nameToPath path) buildConfig.src) kernels)
  );
  srcSet = fileset.unions (map (nameToPath path) extSrc);
in
fileset.toSource {
  root = path;
  fileset = fileset.unions [
    kernelsSrc
    srcSet
    pySrcSet
    pyTestsSet
  ];
}
