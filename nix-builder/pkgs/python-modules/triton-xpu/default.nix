{
  callPackage,
}:

let
  generic = callPackage ./generic.nix { };
in
{
  triton-xpu_3_6_0 = generic {
    version = "3.6.0";
    url = "https://download.pytorch.org/whl/triton_xpu-3.6.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-b2i0GTEJkACA6I8I7jXV/ifboM91BOu+7HS/6sBmrJU=";
  };

  triton-xpu_3_7_0 = generic {
    version = "3.7.0";
    url = "https://download.pytorch.org/whl/triton_xpu-3.7.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-CMjUOygx+vnWeZSA3ytF3eWBAiV669gQ0Hos4YzU5d8=";
  };
}
