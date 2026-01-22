args@{
  pkgs,

  name,

  # Attribute set with names to paths.
  namePaths,

  preferLocalBuild ? true,
  allowSubstitutes ? false,
}:
let
  inherit (pkgs) lib;
  args_ = removeAttrs args [
    "name"
    "pkgs"
    "namePaths"
  ];
  # Iterating over pairs in bash sucks, so let's generate
  # the commands in Nix instead.
  copyPath = path: pkg: ''
    mkdir -p ${placeholder "out"}/${path}
    cp -r ${pkg}/* ${placeholder "out"}/${path}
  '';
  prelude = ''
    mkdir -p ${placeholder "out"}
  '';
in
pkgs.runCommand name args_ (
  prelude + lib.concatStringsSep "\n" (lib.mapAttrsToList copyPath namePaths)
)
