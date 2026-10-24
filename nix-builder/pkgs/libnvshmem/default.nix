{
  buildRedist,
  cuda_cudart,
  cudaAtLeast,
  gdrcopy,
  hwloc,
  lib,
  libfabric,
  libpciaccess,
  libxml2_13,
  mpi,
  pmix,
  rdma-core,
  ucx,
}:

let
  manifest = lib.importJSON ./redistrib_3.8.0.json;
in
buildRedist {
  redistName = "nvshmem";
  pname = "libnvshmem";
  release = manifest.libnvshmem;

  outputs = [
    "out"
    "samples"
  ];

  buildInputs = [
    cuda_cudart
    hwloc
    libfabric
    mpi
    pmix
    rdma-core
    ucx
  ]
  ++ lib.optionals (cudaAtLeast "13.0") [
    libpciaccess
    libxml2_13
  ];

  # Libraries that are dlopened by soname and can therefore not be found by
  # autoPatchelf.
  appendRunpaths = [
    "${lib.getLib rdma-core}/lib"
    "${lib.getLib gdrcopy}/lib"
  ];

  autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];

  allowFHSReferences = true;

  postInstall = ''
    moveToOutput "bin/examples" "''${!outputSamples:?}"
    moveToOutput "bin/perftest" "''${!outputSamples:?}"

    # Bitcode device libraries (~250 MB), only usable with clang. The source
    # build in nixpkgs does not provide them either. Also drop the depfiles.
    rm -v "$out"/lib/*.bc "$out"/lib/*.d
  '';

  meta = {
    description = "Parallel programming interface for NVIDIA GPUs based on OpenSHMEM";
    homepage = "https://developer.nvidia.com/nvshmem";
    license = lib.licenses.nvidiaCudaRedist;
  };
}
