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

  triton-xpu_3_8_0 = generic {
    version = "3.8.0";
    url = "https://huggingface.co/buckets/danieldk/pytorch-rc/resolve/2.14.0/rc6/triton_xpu-3.8.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-Yg7PawctmULzjsvM2APnFpZc6dDWo3Jx2GvtK4U0T04=";
  };
}
