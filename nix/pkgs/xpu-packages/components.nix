final: prev:

# Create a package for all components in the Intel oneAPI Base Kit metadata.
prev.lib.mapAttrs (
  pname: metadata:
  prev.callPackage ./generic.nix {
    inherit pname;
    inherit (metadata) components deps version;
  }
) prev.packageMetadata
