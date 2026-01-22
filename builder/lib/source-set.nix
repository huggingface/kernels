{ lib }:

path:

let
  inherit (lib) fileset;
  readToml = path: builtins.fromTOML (builtins.readFile path);
  readBuildConfig = path: readToml (path + "/build.toml");
  buildConfig = readBuildConfig path;
  nameToPath = path: name: path + "/${name}";
  kernels = buildConfig.kernel or { };
  extConfig = buildConfig.torch or { };
  pyExt =
    extConfig.pyext or [
      "py"
      "pyi"
    ];
  pyFilter = file: builtins.any (ext: file.hasExt ext) pyExt;
  extSrc = extConfig.src or [ ] ++ [ "build.toml" ];
  pySrcSet = fileset.fileFilter pyFilter (path + "/torch-ext");
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
  ];
}
