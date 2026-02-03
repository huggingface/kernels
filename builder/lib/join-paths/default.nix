args@{
  pkgs,

  name,

  # Package paths.
  contents,

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
  copyPkg = pkg: ''
    cp -r ${pkg}/* ${placeholder "out"}/
  '';
  prelude = ''
    mkdir -p ${placeholder "out"}
  '';
in
pkgs.runCommand name args_ (prelude + lib.concatStringsSep "\n" (builtins.map copyPkg contents))
