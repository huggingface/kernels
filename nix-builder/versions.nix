[
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
    # ptxas 13.0 segfaults on aarch64 compiling large sm90 kernels (flash-attn3).
    ptxasVersion.aarch64-linux = "13.4";
    systems = [
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
    rocmVersion = "7.0";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.10";
    rocmVersion = "7.1";
    systems = [ "x86_64-linux" ];
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
    cudaVersion = "12.6";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
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
    # ptxas 13.0 segfaults on aarch64 compiling large sm90 kernels (flash-attn3).
    ptxasVersion.aarch64-linux = "13.4";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
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
  }
  {
    torchVersion = "2.11";
    tpu = true;
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
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
    # ptxas 13.0 segfaults on aarch64 compiling large sm90 kernels (flash-attn3).
    ptxasVersion.aarch64-linux = "13.4";
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

  {
    torchVersion = "2.13";
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
    torchVersion = "2.13";
    cudaVersion = "12.6";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.13";
    cudaVersion = "13.0";
    # ptxas 13.0 segfaults on aarch64 compiling large sm90 kernels (flash-attn3).
    ptxasVersion.aarch64-linux = "13.4";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
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
  {
    torchVersion = "2.13";
    rocmVersion = "7.1";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.13";
    rocmVersion = "7.2";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.13";
    xpuVersion = "2026.0.0";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
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
    # ptxas 13.0 segfaults on aarch64 compiling large sm90 kernels (flash-attn3).
    ptxasVersion.aarch64-linux = "13.4";
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
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.14";
    xpuVersion = "2026.1.0";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }

  {
    torchVersion = "2.15";
    cpu = true;
    systems = [
      "aarch64-darwin"
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }

  {
    torchVersion = "2.15";
    cudaVersion = "13.2";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.15";
    cudaVersion = "13.4";
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.15";
    metal = true;
    systems = [ "aarch64-darwin" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.15";
    rocmVersion = "7.14";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.15";
    rocmVersion = "10.0";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
  {
    torchVersion = "2.15";
    xpuVersion = "2026.1.2";
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
    tvmFfiVersion = "0.1";
  }
]
