{ lib }:

let
  inherit (lib) versions;
in
{
  flattenVersion =
    version: lib.replaceStrings [ "." ] [ "" ] (versions.majorMinor (versions.pad 2 version));
}
