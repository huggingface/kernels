{
  callPackage,
}:

let
  generic = callPackage ./generic.nix { };
in
{
  triton_3_6_0 = generic {
    version = "3.6.0";
    url = "https://download.pytorch.org/whl/triton-3.6.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-AHUDn/J3ZUgAg7GhCZmb8nEQ81QvH5+tlfD5Blo22nk=";
  };

  triton_3_7_0 = generic {
    version = "3.7.0";
    url = "https://download.pytorch.org/whl/test/triton-3.7.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-yvYg5j6eBA83/eq+Pe1fsvzVYv78SRZp6s2DFWLuNwk=";
  };
}
