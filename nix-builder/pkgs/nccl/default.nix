{
  cudaMajorVersion,
  fetchurl,
  lib,
  stdenv,

  # Native build inputs
  autoPatchelfHook,
  unzip,

  # Build inputs
  gdrcopy,
  rdma-core,
}:

let
  version = "2.32.3";

  wheels = {
    x86_64-linux = {
      cuda12 = {
        url = "https://files.pythonhosted.org/packages/f3/f8/9dce3698eed6fae28078eb38f748feb1665767693d8017c099de9fbf0b08/nvidia_nccl_cu12-2.32.3-py3-none-manylinux_2_27_x86_64.whl";
        sha256 = "bb94b2348e4861d5e8aa00b36c3d12d9bdf9c6369b82cbb723de0b582280a21c";
      };
      cuda13 = {
        url = "https://files.pythonhosted.org/packages/5b/29/6b277e63c92d91f9cb4d1a3a554e148983de39d54baa652bb52c798af78e/nvidia_nccl_cu13-2.32.3-py3-none-manylinux_2_27_x86_64.whl";
        sha256 = "1459723080ac889d73a26edfa3e04383a7928ab31ac8f0ec43b3ea9548b04ff3";
      };
    };
    aarch64-linux = {
      cuda12 = {
        url = "https://files.pythonhosted.org/packages/46/cb/702524922e4f64e42e8c299a26c7d794bce50dee2f5f54758f4d1e162305/nvidia_nccl_cu12-2.32.3-py3-none-manylinux_2_27_aarch64.whl";
        sha256 = "061af42ae1044816820e16b872075e243d1ac0656f51860fbc8e106c257aa83a";
      };
      cuda13 = {
        url = "https://files.pythonhosted.org/packages/58/0a/c29c302036a06d27dd732588f3fd8ee89b1f7b0087e38c668ae7be8ff7b2/nvidia_nccl_cu13-2.32.3-py3-none-manylinux_2_27_aarch64.whl";
        sha256 = "a5bee92b2f4af218c109f8d221c3c9adcc752b94ae0ffcb0ed5abf9341a7724c";
      };
    };
  };

  inherit (stdenv.hostPlatform) system;

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
    unzip
  ];

  buildInputs = [ stdenv.cc.cc.lib ];

  autoPatchelfFlags = [ "--keep-libc" ];

  # Add rpaths of dlopen'ed libraries.
  appendRunpaths = [
    "${lib.getLib rdma-core}/lib"
    "${lib.getLib gdrcopy}/lib"
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
