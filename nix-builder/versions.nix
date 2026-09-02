[
  {
    torchVersion = "2.11";
    cpu = true;
    systems = [
      "x86_64-linux"
    ];
    bundleBuild = false;
  }
  {
    torchVersion = "2.11";
    tpu = true;
    systems = [ "x86_64-linux" ];
    # Excluded from the shared build cache: torch_tpu's autoPatchelfHook
    # step produces an output that is itself a modified copy of Google's
    # gated libtpu/torch_tpu wheels, so it can't be published publicly.
    bundleBuild = false;
  }

  {
    torchVersion = "2.12";
    cpu = true;
    systems = [
      "aarch64-darwin"
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.12";
    cudaVersion = "12.6";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.12";
    cudaVersion = "13.0";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.12";
    cudaVersion = "13.2";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.12";
    metal = true;
    systems = [ "aarch64-darwin" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.12";
    rocmVersion = "7.1";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.12";
    rocmVersion = "7.2";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.12";
    xpuVersion = "2025.3.2";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }

  {
    torchVersion = "2.13";
    cpu = true;
    systems = [
      "aarch64-darwin"
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.13";
    cudaVersion = "12.6";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.13";
    cudaVersion = "13.0";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.13";
    cudaVersion = "13.2";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.13";
    metal = true;
    systems = [ "aarch64-darwin" ];
    bundleBuild = true;
  }
  # Broken: https://github.com/ROCm/ROCm/issues/6322
  {
    torchVersion = "2.13";
    rocmVersion = "7.1";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.13";
    rocmVersion = "7.2";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.13";
    xpuVersion = "2026.0.0";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }

  {
    torchVersion = "2.14";
    cpu = true;
    systems = [
      "aarch64-darwin"
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.14";
    cudaVersion = "12.6";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.14";
    cudaVersion = "13.0";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.14";
    cudaVersion = "13.2";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  # Since 13.4 is a prerelease, not yet available on:
  # https://developer.download.nvidia.com/compute/cuda/redist/
  #{
  #  torchVersion = "2.14";
  #  cudaVersion = "13.4";
  #  systems = [
  #    "x86_64-linux"
  #    "aarch64-linux"
  #  ];
  #  bundleBuild = true;
  #  tvmFfiVersion = "0.1";
  #}
  {
    torchVersion = "2.14";
    metal = true;
    systems = [ "aarch64-darwin" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.14";
    rocmVersion = "7.2";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.14";
    rocmVersion = "7.14";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.14";
    xpuVersion = "2026.1.0";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
]
