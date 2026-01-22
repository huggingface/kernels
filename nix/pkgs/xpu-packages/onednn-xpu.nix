{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  ninja,
  gcc,
  setupXpuHook,
  oneapi-torch-dev,
}:

let
  dpcppVersion = oneapi-torch-dev.version;
  oneDnnVersions = {
    "2025.1" = {
      version = "3.8.1";
      hash = "sha256-x4leRd0xPFUygjAv/D125CIXn7lYSyzUKsd9IDh/vCc=";
    };
    "2025.2" = {
      version = "3.9.1";
      hash = "sha256-DbLW22LgG8wrBNMsxoUGlacHLcfIBwqyiv+HOmFDtxc=";
    };
    "2025.3" = {
      version = "3.10.2";
      hash = "sha256-/e57voLBNun/2koTF3sEb0Z/nDjCwq9NJVk7TaTSvMY=";
    };
  };
  oneDnnVersion =
    oneDnnVersions.${lib.versions.majorMinor dpcppVersion}
    or (throw "Unsupported DPC++ version: ${dpcppVersion}");
in
stdenv.mkDerivation (finalAttrs: {
  pname = "onednn-xpu";
  inherit (oneDnnVersion) version;

  src = fetchFromGitHub {
    owner = "oneapi-src";
    repo = "oneDNN";
    tag = "v${oneDnnVersion.version}";
    inherit (oneDnnVersion) hash;
  };

  nativeBuildInputs = [
    cmake
    ninja
    setupXpuHook
    oneapi-torch-dev
  ];

  cmakeFlags = [
    "-DCMAKE_C_COMPILER=icx"
    "-DCMAKE_CXX_COMPILER=icpx"
    "-DDNNL_GPU_RUNTIME=SYCL"
    "-DDNNL_CPU_RUNTIME=THREADPOOL"
    "-DDNNL_BUILD_TESTS=OFF"
    "-DDNNL_BUILD_EXAMPLES=OFF"
    "-DONEDNN_BUILD_GRAPH=ON"
    "-DDNNL_LIBRARY_TYPE=STATIC"
  ];

  postInstall = ''
    if [ "${finalAttrs.version}" != "3.7.1" ]; then
      cp -rn "$src/third_party/level_zero" "$out/include/"
    else
      cp -rn "$src/src/gpu/intel/sycl/l0/level_zero" "$out/include/"
    fi
  '';
})
