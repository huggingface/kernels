{
  callPackage,
}:

let
  generic = callPackage ./generic.nix { };
in
{
  triton-xpu_3_7_1 = generic {
    version = "3.7.1";
    url = "https://download-r2.pytorch.org/whl/triton_xpu-3.7.1-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-69d2XCnsKJ3OQSWumuABCUUyTRbDLOCGbBfGmyc7V/8=";
  };

  triton-xpu_3_7_2 = generic {
    version = "3.7.2";
    url = "https://download.pytorch.org/whl/triton_xpu-3.7.2-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-Vgtk8KGUCN7jmx49CN9hT8m7KUqTCkP/E2HRvhLwyYA=";
  };

  triton-xpu_3_8_0 = generic {
    version = "3.8.0";
    url = "https://download.pytorch.org/whl/triton_xpu-3.8.0-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-r8W70Ter/+iBD4uLfcZU/zdJe7KLspPT/qag5PH7qwk=";
  };
}
