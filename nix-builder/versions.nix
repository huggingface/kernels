[
  {
    torchVersion = "2.9";
    cudaVersion = "12.9";
    systems = [
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

  {
    torchVersion = "2.11";
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
    torchVersion = "2.11";
    cudaVersion = "12.6";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.11";
    cudaVersion = "12.8";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.11";
    cudaVersion = "13.0";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.11";
    metal = true;
    systems = [ "aarch64-darwin" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.11";
    rocmVersion = "7.1";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.11";
    rocmVersion = "7.2";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.11";
    xpuVersion = "2025.3.2";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
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
  }
  {
    torchVersion = "2.12";
    cudaVersion = "12.6";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.12";
    cudaVersion = "13.0";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
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
  }

]
