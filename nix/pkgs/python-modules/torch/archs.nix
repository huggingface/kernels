{
  "2.8" = {
    # https://github.com/pytorch/pytorch/blob/release/2.8/.ci/manywheel/build_cuda.sh
    capsPerCudaVersion = {
      "12.9" = [
        "7.0"
        "7.5"
        "8.0"
        "8.6"
        "9.0"
        "10.0"
        "12.0"
      ];
      "12.8" = [
        "7.0"
        "7.5"
        "8.0"
        "8.6"
        "9.0"
        "10.0"
        "12.0"
      ];
      "12.6" = [
        "5.0"
        "6.0"
        "7.0"
        "7.5"
        "8.0"
        "8.6"
        "9.0"
      ];
      # Not a supported upstream configuration, but keep it around for
      # builds that fail on newer CUDA versions.
      "12.4" = [
        "5.0"
        "6.0"
        "7.0"
        "7.5"
        "8.0"
        "8.6"
        "9.0"
      ];
    };
    # https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/.ci/docker/manywheel/build.sh#L82
    supportedTorchRocmArchs = [
      "gfx900"
      "gfx906"
      "gfx908"
      "gfx90a"
      "gfx942"
      "gfx1030"
      "gfx1100"
      "gfx1101"
      "gfx1102"
      "gfx1200"
      "gfx1201"
    ];
  };

  "2.9" = {
    # https://github.com/pytorch/pytorch/blob/release/2.9/.ci/manywheel/build_cuda.sh
    capsPerCudaVersion = {
      "13.0" = [
        "7.5"
        "8.0"
        "8.6"
        "9.0"
        "10.0"
        "12.0"
      ];
      # NOTE: 12.9 does not seem to be in RC builds, check if needed for final release.
      #       https://download.pytorch.org/whl/test/torch/
      "12.9" = [
        "7.0"
        "7.5"
        "8.0"
        "8.6"
        "9.0"
        "10.0"
        "12.0"
      ];
      "12.8" = [
        "7.0"
        "7.5"
        "8.0"
        "8.6"
        "9.0"
        "10.0"
        "12.0"
      ];
      "12.6" = [
        "5.0"
        "6.0"
        "7.0"
        "7.5"
        "8.0"
        "8.6"
        "9.0"
      ];
    };

    supportedTorchRocmArchs = [
      # https://github.com/pytorch/pytorch/blob/21fec65781bebe867faf209f89bb687ffd236ca4/.ci/docker/manywheel/build.sh#L92
      "gfx900"
      "gfx906"
      "gfx908"
      "gfx90a"
      "gfx942"
      "gfx1030"
      "gfx1100"
      "gfx1101"
      "gfx1102"
      "gfx1200"
      "gfx1201"
    ];
  };

  "2.10" = {
    # https://github.com/pytorch/pytorch/blob/release/2.9/.ci/manywheel/build_cuda.sh
    capsPerCudaVersion = {
      "13.0" = [
        "7.5"
        "8.0"
        "8.6"
        "9.0"
        "10.0"
        "12.0"
      ];
      # NOTE: 12.9 does not seem to be in RC builds, check if needed for final release.
      #       https://download.pytorch.org/whl/test/torch/
      "12.9" = [
        "7.0"
        "7.5"
        "8.0"
        "8.6"
        "9.0"
        "10.0"
        "12.0"
      ];
      "12.8" = [
        "7.0"
        "7.5"
        "8.0"
        "8.6"
        "9.0"
        "10.0"
        "12.0"
      ];
      "12.6" = [
        "5.0"
        "6.0"
        "7.0"
        "7.5"
        "8.0"
        "8.6"
        "9.0"
      ];
    };

    supportedTorchRocmArchs = [
      # https://github.com/pytorch/pytorch/blob/release/2.10/.ci/docker/almalinux/build.sh
      "gfx900"
      "gfx906"
      "gfx908"
      "gfx90a"
      "gfx942"
      "gfx950"
      "gfx1030"
      "gfx1100"
      "gfx1101"
      "gfx1102"
      "gfx1150"
      "gfx1151"
      "gfx1200"
      "gfx1201"
    ];
  };

}
