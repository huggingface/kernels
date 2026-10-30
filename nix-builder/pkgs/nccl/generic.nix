{
  fetchurl,
  lib,
  stdenv,

  # Native build inputs
  autoPatchelfHook,
  unzip,

  # Build inputs
  cudaPackages,
  rdma-core,
}:

# Build NCCL from the nvidia-nccl-cu12/cu13 wheels. `wheels` is an attribute
# set that maps a system and CUDA major version (e.g. `cuda12`) to the wheel
# `url` and `sha256`.
{
  version,
  wheels,
}:

let
  inherit (stdenv.hostPlatform) system;
  inherit (cudaPackages) cudaMajorVersion;

  wheel =
    wheels.${system}."cuda${cudaMajorVersion}"
      or (throw "NCCL ${version} is not available for ${system} and CUDA ${cudaMajorVersion}");
in
stdenv.mkDerivation {
  pname = "nccl";
  inherit version;

  src = fetchurl {
    inherit (wheel) url sha256;
  };

  dontUnpack = true;

  nativeBuildInputs = [
    autoPatchelfHook
    cudaPackages.markForCudatoolkitRootHook
    unzip
  ];

  buildInputs = [ stdenv.cc.cc.lib ];

  autoPatchelfFlags = [ "--keep-libc" ];

  # Add rpaths of dlopen'ed libraries.
  appendRunpaths = [
    "${lib.getLib rdma-core}/lib"
    "${lib.getLib cudaPackages.gdrcopy}/lib"
  ];

  installPhase = ''
    runHook preInstall

    unzip -q "$src" 'nvidia/nccl/*' -d unpacked
    mkdir -p "$out"
    cp -r unpacked/nvidia/nccl/include unpacked/nvidia/nccl/lib "$out"/

    ln -s libnccl.so.2 "$out/lib/libnccl.so"

    runHook postInstall
  '';

  meta = {
    description = "NVIDIA Collective Communications Library (NCCL)";
    homepage = "https://developer.nvidia.com/nccl";
    license = lib.licenses.bsd3;
    platforms = [
      "aarch64-linux"
      "x86_64-linux"
    ];
    sourceProvenance = [ lib.sourceTypes.binaryNativeCode ];
  };
}
