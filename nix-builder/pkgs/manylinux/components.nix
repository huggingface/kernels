final: prev:

prev.lib.mapAttrs (
  pname: metadata:
  final.callPackage ./generic.nix {
    inherit pname;
    inherit (metadata) components deps version;
    manylinuxPackages = final;
  }
) prev.packageMetadata
