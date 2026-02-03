[
  {
    torchVersion = "2.9";
    xpuVersion = "2025.2.1";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.9";
    cudaVersion = "12.6";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.9";
    cudaVersion = "12.8";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.9";
    cudaVersion = "13.0";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.9";
    rocmVersion = "6.3.4";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.9";
    rocmVersion = "6.4.2";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.9";
    metal = true;
    systems = [ "aarch64-darwin" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.9";
    cpu = true;
    systems = [
      "aarch64-darwin"
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }

  {
    torchVersion = "2.10";
    cudaVersion = "12.6";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.10";
    cudaVersion = "12.8";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.10";
    cudaVersion = "13.0";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.10";
    rocmVersion = "7.0";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.10";
    rocmVersion = "7.1";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.10";
    cpu = true;
    systems = [
      "aarch64-darwin"
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.10";
    metal = true;
    systems = [ "aarch64-darwin" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.10";
    xpuVersion = "2025.3.1";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
]
