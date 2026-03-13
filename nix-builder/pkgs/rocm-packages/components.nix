final: prev:

# Create a package for all components in the ROCm runfile metadata.
prev.lib.mapAttrs (
  pname: metadata:
  prev.callPackage ./generic.nix {
    inherit pname;
    inherit (metadata) components deps version;
    rocmPackages = final;
  }
) prev.packageMetadata
