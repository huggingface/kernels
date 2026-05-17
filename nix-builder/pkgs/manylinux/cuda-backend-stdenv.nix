{ pkgs }:

final: prev: {
  cudaBackendStdenv = import ../cuda/backendStdenv {
    inherit (pkgs)
      _cuda
      config
      lib
      stdenvAdapters
      ;
    inherit (pkgs.cudaPackages) cudaMajorMinorVersion;
    # Allow auto-selection to use manylinux gcc versions.
    pkgs = final;
    stdenv = final.stdenv;
  };
}
