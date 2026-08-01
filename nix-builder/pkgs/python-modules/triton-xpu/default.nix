{
  callPackage,
}:

let
  generic = callPackage ./generic.nix { };
in
{
  triton-xpu_3_7_1 = generic {
    version = "3.7.1";
    url = "https://download-r2.pytorch.org/whl/triton_xpu-3.7.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-5wGjHvoDNHdfNXyYcW84IXdaqUQhn3iI4Twt/i2qvio=";
  };

  triton-xpu_3_7_2 = generic {
    version = "3.7.2";
    url = "https://download.pytorch.org/whl/triton_xpu-3.7.2-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-jMwVkt3aI6sunRQ9orS2nrQIIRcsW9S8TiYjI9x/UXI=";
  };

}
